import numpy as np
from adaptive_filter.scale_relative_sensitivity import apply_relative_sensitivity_to_fft_dict
from adaptive_filter.tools import normalize_minmax

class InitialFilterParams:
    def __init__(
        self,
        n_filters=128,
        contrast_weight=0.3,
        variance_weight=0.5,
        rel_variance_weight=0.1,
        energy_weight=0.2,
        eps=1e-12,
        bin_start=0,
        bin_end=512
    ):
        self.n_filters = n_filters

        self.rel_variance_weight = rel_variance_weight
        self.contrast_weight = contrast_weight
        self.variance_weight = variance_weight
        self.energy_weight = energy_weight

        self.eps = eps
        self.bin_start = bin_start
        self.bin_end = bin_end
        self.k_per_bin = self.generate_log_decreasing_K_per_bin()


        self.bins = None
        self.edges = None
        self.filters = None

    # ============================
    # Core API
    # ============================


    def generate_log_decreasing_K_per_bin(
        self,
        n_bins=512,
        K_max=8,
        K_min=2,
        bin_ref_low=1,      # avoid log(0)
        bin_ref_high=511
    ):
        """
        Returns:
            K_per_bin: np.ndarray of shape (n_bins,)
                    integer local contrast radius per bin
        """

        bins = np.arange(n_bins)

        # avoid log(0) at DC
        bins_safe = np.maximum(bins, bin_ref_low)

        log_b = np.log2(bins_safe)

        log_low  = np.log2(bin_ref_low)
        log_high = np.log2(bin_ref_high)

        # normalize to [0, 1]
        t = (log_b - log_low) / (log_high - log_low)
        t = np.clip(t, 0.0, 1.0)

        # invert: low bins -> large K
        K = K_max - t * (K_max - K_min)

        return np.round(K).astype(int)

    def fit(self, fft_dict):
        """
        fft_dict: dict[str, np.ndarray(n_bins)]  (linear magnitude)
        """

        # --------------------------------------------------
        # 0. Dict -> matrix
        # --------------------------------------------------
        keys = list(fft_dict.keys())
        fft = np.stack([fft_dict[k][self.bin_start:self.bin_end] for k in keys], axis=0)  # (N, B)

        fft_db = 20 * np.log10(fft + self.eps)

        N, B = fft_db.shape

        # --------------------------------------------------
        # 1. Mean spectrum & residuals
        # --------------------------------------------------
        mean_spec = fft_db.mean(axis=0)
        residuals = fft_db - mean_spec

        # --------------------------------------------------
        # 2. Variance between sounds
        # --------------------------------------------------
        between_var = residuals.var(axis=0)

        # --------------------------------------------------
        # 3. Local spectral contrast
        # --------------------------------------------------
        contrast_all = []

        for i in range(N):
            spec = fft_db[i]
            K = self.k_per_bin[i+self.bin_start]
            local_median = np.zeros(B)
            for b in range(B):
                lo = max(0, b - K)
                hi = min(B, b + K + 1)
                local_median[b] = np.median(spec[lo:hi])

            contrast_all.append(spec - local_median)

        contrast_all = np.stack(contrast_all, axis=0)
        local_contrast = contrast_all.mean(axis=0)

        # --------------------------------------------------
        # 4. Relative variance
        # --------------------------------------------------
        relative_var = between_var/mean_spec

        # --------------------------------------------------
        # 5. Energy (equal loudness)
        # --------------------------------------------------
        loud_dict = apply_relative_sensitivity_to_fft_dict(
            fft_dict,
            eps=self.eps
        )
        X_loud = np.stack([loud_dict[k][self.bin_start:self.bin_end] for k in keys], axis=0)
        energy = X_loud.mean(axis=0)

        # --------------------------------------------------
        # 6. Normalization
        # --------------------------------------------------
        between_var_n   = normalize_minmax(between_var, self.eps)
        contrast_n      = normalize_minmax(np.abs(local_contrast), self.eps)
        relative_var_n = normalize_minmax(relative_var, self.eps)
        energy_n        = normalize_minmax(energy, self.eps)

        # --------------------------------------------------
        # 7. Information density
        # --------------------------------------------------
        info_density = (
            self.variance_weight * between_var_n +
            self.contrast_weight * contrast_n +
            self.rel_variance_weight * relative_var_n +
            self.energy_weight * energy_n
        )

        info_density = np.maximum(info_density, self.eps)
        info_density /= info_density.sum()
        # --------------------------------------------------
        # 6. Build edges
        # --------------------------------------------------
        self.edges = self.adaptive_equal_info_edges(info_density)

        return self


    def save(self, path):
        if self.edges is None:
            raise RuntimeError("Nothing to save. Call fit() first.")

        np.savez(
            path,
            edges=np.array(self.edges, dtype=np.int32),
            n_filters=self.n_filters,
            contrast_weight=self.contrast_weight,
            rel_variance_weight=self.rel_variance_weight,
            variance_weight=self.variance_weight,
            energy_weight=self.energy_weight,
            eps=self.eps,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)

        obj = cls(
            n_filters=int(data["n_filters"]),
            overlap=float(data["overlap"]),
            contrast_weight=float(data["contrast_weight"]),
            rel_variance_weight=float(data["rel_variance_weight"]),
            variance_weight=float(data["variance_weight"]),
            energy_weight=float(data["energy_weight"]),
            eps=float(data["eps"]),
        )

        obj.edges = [tuple(b) for b in data["edges"]]
        return obj
    
    def adaptive_equal_info_edges(self, density):
        """
        Build bin edges such that each of self.n_filters ranges carries
        equal information mass, allowing per-bin overflow.

        Overflow means: if a single bin has more mass than 1/n_filters,
        the excess is carried into subsequent filters.
        Alerts if any bin exceeds 1/n_filters.

        Returns:
            edges: (n_filters + 1,) int array of bin edges (local indices)
        """
        density = np.asarray(density, dtype=float)
        B = len(density)

        if self.n_filters < 2:
            raise ValueError("n_filters must be >= 2")

        # -----------------------------
        # Normalize density
        # -----------------------------
        total_mass = density.sum()
        if total_mass <= 0:
            raise ValueError("Density has zero total mass")
        density = density / total_mass

        mass_per_filter = 1.0 / self.n_filters

        # -----------------------------
        # Check for overflow bins
        # -----------------------------
        overflow_bins = np.where(density > mass_per_filter)[0]
        if len(overflow_bins) > 0:
            print(f"WARNING: {len(overflow_bins)} bin(s) exceed 1/filter mass ({mass_per_filter:.4f}): {overflow_bins}")

        edges = [0]  # first bin
        current_mass = 0.0
        filters_assigned = 0
        for b in range(B):
            bin_mass = density[b]
            while bin_mass + current_mass > mass_per_filter and filters_assigned < self.n_filters - 1:
                # assign remaining mass to current filter
                remaining = mass_per_filter - current_mass
                bin_mass -= remaining
                edges.append(b + 1)  # end of current filter
                filters_assigned += 1
                current_mass = 0.0
            current_mass += bin_mass

        # Last filter always ends at last bin
        edges.append(B)
        edges = np.array(edges, dtype=int)

        # Safety: ensure monotonicity
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1
        edges = np.clip(edges, 0, B)

        edges = edges + self.bin_start

        return edges
