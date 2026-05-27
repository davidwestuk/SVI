from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike

# Type alias for flexible scalar/array input
Strikes = float | list[float] | np.ndarray


@dataclass
class SVIJumpWings:
    """
    SVI Jump Wings (natural) parameterization of the implied variance surface.

    Parameters correspond to physical characteristics of w(K, T) at a single maturity T:
      w0         : ATM implied variance       (w(F, T))
      w1         : ATM variance skew          (∂w/∂k at k=0)
      w2         : ATM variance curvature     (∂²w/∂k² at k=0)
      beta_minus : left-wing (put) slope  — lim_{k→-∞} w/k = b(ρ - 1)
      beta_plus  : right-wing (call) slope — lim_{k→+∞} w/k = b(ρ + 1)
      F          : forward price
      T          : maturity in years

    Reference: Gatheral (2004), Gatheral & Jacquier (2013),
               Numerix SVI Volatility Surface (2024).
    """
    w0: float
    w1: float
    w2: float
    beta_minus: float
    beta_plus: float
    F: float
    T: float

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_array(k: Strikes) -> tuple[np.ndarray, bool]:
        """
        Normalise k to a float64 ndarray.
        Returns (array, scalar_input) so callers can unwrap scalars on output.
        """
        scalar = isinstance(k, (int, float))
        arr = np.atleast_1d(np.asarray(k, dtype=np.float64))
        return arr, scalar

    @staticmethod
    def _maybe_scalar(arr: np.ndarray, scalar: bool) -> float | np.ndarray:
        """Return a plain float if the original input was scalar, else ndarray."""
        return float(arr[0]) if scalar else arr

    # ------------------------------------------------------------------ #
    # Derived original parameters  (Equations 2.7–2.11, Numerix 2024)    #
    # ------------------------------------------------------------------ #

    @property
    def b(self) -> float:
        """b = (β+ − β−) / 2"""
        return (self.beta_plus - self.beta_minus) / 2.0

    @property
    def rho(self) -> float:
        """ρ = (β+ + β−) / (β+ − β−)"""
        denom = self.beta_plus - self.beta_minus
        if denom == 0.0:
            raise ValueError("β+ and β− must not be equal (b would be zero).")
        return (self.beta_plus + self.beta_minus) / denom

    @property
    def sigma(self) -> float:
        b, rho, w1, w2 = self.b, self.rho, self.w1, self.w2
        inner = 1.0 - (rho - w1 / b) ** 2
        return math.sqrt(b ** 2 * inner ** 3 / w2 ** 2)

    @property
    def m(self) -> float:
        b, rho, w1, w2 = self.b, self.rho, self.w1, self.w2
        inner = 1.0 - (rho - w1 / b) ** 2
        m_sq = b ** 2 * inner ** 2 * (rho - w1 / b) ** 2 / w2 ** 2
        sign = math.copysign(1.0, rho - w1 / b)
        return sign * math.sqrt(m_sq)

    @property
    def a(self) -> float:
        b, rho, m, sigma = self.b, self.rho, self.m, self.sigma
        return self.w0 - b * (-rho * m + math.sqrt(sigma ** 2 + m ** 2))

    # ------------------------------------------------------------------ #
    # Core formula and derivatives                                         #
    # ------------------------------------------------------------------ #

    def implied_variance(self, K: Strikes) -> float | np.ndarray:
        """
        Implied Black-Scholes total variance w(K).
        w = a + b[ρ(k−m) + sqrt(σ²+(k−m)²)], where k = log(K/F)

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.
        """
        K_arr, scalar = self._to_array(K)
        k_arr = np.log(K_arr / self.F)
        a, b, m, sigma, rho = self.a, self.b, self.m, self.sigma, self.rho
        d = k_arr - m
        result = a + b * (rho * d + np.sqrt(sigma ** 2 + d ** 2))
        return self._maybe_scalar(result, scalar)

    def dw_dk(self, K: Strikes) -> float | np.ndarray:
        """
        First derivative ∂w/∂k evaluated at k = log(K/F).

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.
        """
        K_arr, scalar = self._to_array(K)
        k_arr = np.log(K_arr / self.F)
        b, m, sigma, rho = self.b, self.m, self.sigma, self.rho
        d = k_arr - m
        result = b * (rho + d / np.sqrt(sigma ** 2 + d ** 2))
        return self._maybe_scalar(result, scalar)

    def d2w_dk2(self, K: Strikes) -> float | np.ndarray:
        """
        Second derivative ∂²w/∂k² evaluated at k = log(K/F).

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.
        """
        K_arr, scalar = self._to_array(K)
        k_arr = np.log(K_arr / self.F)
        b, m, sigma = self.b, self.m, self.sigma
        d = k_arr - m
        result = b * sigma ** 2 / (sigma ** 2 + d ** 2) ** 1.5
        return self._maybe_scalar(result, scalar)

    def implied_vol(self, K: Strikes) -> float | np.ndarray:
        """
        Black-Scholes implied volatility σ_BS(K) = sqrt(w(K) / T).

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.

        Raises
        ------
        ValueError if any element of w(K) is negative.
        """
        w = self.implied_variance(K)
        w_arr = np.atleast_1d(np.asarray(w, dtype=np.float64))
        if np.any(w_arr < 0):
            bad = w_arr[w_arr < 0]
            raise ValueError(f"Negative implied variance(s) encountered: {bad}")
        result = np.sqrt(w_arr / self.T)
        scalar = isinstance(K, (int, float))
        return self._maybe_scalar(result, scalar)

    # ------------------------------------------------------------------ #
    # Butterfly arbitrage: g(k) function                                  #
    # ------------------------------------------------------------------ #

    def g(self, K: Strikes) -> float | np.ndarray:
        """
        The Gatheral-Jacquier g(K) function (Lemma 2.2, Gatheral & Jacquier 2013).

        A slice is free of butterfly arbitrage if and only if g(K) >= 0 for all K.
        Equivalently, g(K) >= 0 is necessary and sufficient for the risk-neutral
        density to be non-negative at K.

        Definition (in terms of k = log(K/F)):
            g(K) = (1 - k*w'/(2w))²  -  (w'/2)² * (1/w + 1/4)  +  w''/2

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.
        """
        K_arr, scalar = self._to_array(K)
        k_arr = np.log(K_arr / self.F)
        w   = np.atleast_1d(np.asarray(self.implied_variance(K_arr), dtype=np.float64))
        wp  = np.atleast_1d(np.asarray(self.dw_dk(K_arr),            dtype=np.float64))
        wpp = np.atleast_1d(np.asarray(self.d2w_dk2(K_arr),          dtype=np.float64))

        term1 = (1.0 - k_arr * wp / (2.0 * w)) ** 2
        term2 = (wp ** 2 / 4.0) * (1.0 / w + 0.25)
        term3 = wpp / 2.0

        result = term1 - term2 + term3
        return self._maybe_scalar(result, scalar)

    # ------------------------------------------------------------------ #
    # Risk-neutral density                                                 #
    # ------------------------------------------------------------------ #

    def risk_neutral_density(self, K: Strikes) -> float | np.ndarray:
        """
        Risk-neutral probability density over log-forward moneyness k = log(K/F).

        From Gatheral & Jacquier (2013), Lemma 2.2:

            p(K) = g(K) / sqrt(2π w(K)) · exp(−d₂(K)² / 2)

        where:
            w(K)  = total implied variance (σ²_BS · T), from implied_variance(K)
            d₂(K) = −k / sqrt(w(K)) − sqrt(w(K)) / 2,  k = log(K/F)
            g(K)  = Gatheral-Jacquier function, from g(K)

        T does not appear separately because w(K) is already total variance.
        The density integrates to 1 over k when the slice is butterfly-arbitrage-free.

        Parameters
        ----------
        K : strike price(s). Accepts float, list[float], or np.ndarray.

        Returns
        -------
        float if K was scalar, np.ndarray otherwise.
        """
        K_arr, scalar = self._to_array(K)
        k_arr = np.log(K_arr / self.F)

        w      = np.atleast_1d(np.asarray(self.implied_variance(K_arr), dtype=np.float64))
        g_vals = np.atleast_1d(np.asarray(self.g(K_arr),                dtype=np.float64))

        sqrt_w = np.sqrt(w)
        d2     = -k_arr / sqrt_w - 0.5 * sqrt_w
        result = g_vals * np.exp(-0.5 * d2 ** 2) / (sqrt_w * np.sqrt(2.0 * np.pi))

        return self._maybe_scalar(result, scalar)

    def is_butterfly_arbitrage_free(
        self,
        K_grid: np.ndarray | None = None,
        tol: float = 0.0,
    ) -> bool:
        """
        Returns True if g(K) >= tol for all K in K_grid (Lemma 2.2).

        Parameters
        ----------
        K_grid : array of strikes to check. Defaults to F·exp([-5, 5]) with 2001 points.
        tol    : numerical tolerance; set slightly negative to allow for floating-point
                 noise in near-boundary cases.
        """
        if K_grid is None:
            K_grid = self.F * np.exp(np.linspace(-5.0, 5.0, 2001))
        return bool(np.all(self.g(K_grid) >= tol))

    def min_g(
        self,
        K_grid: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Returns (K*, g(K*)) where K* is the strike that minimises g over K_grid.
        Useful as a scalar diagnostic of how close the slice is to butterfly arbitrage.
        """
        if K_grid is None:
            K_grid = self.F * np.exp(np.linspace(-5.0, 5.0, 2001))
        g_vals = self.g(K_grid)
        idx = int(np.argmin(g_vals))
        return float(K_grid[idx]), float(g_vals[idx])

    # ------------------------------------------------------------------ #
    # Strike arbitrage (Numerix 2024, Section 4.2)                        #
    # ------------------------------------------------------------------ #

    def is_strike_arbitrage_free(self, tol: float = 1e-8) -> bool:
        """
        Checks the two strike no-arbitrage conditions (Equations 4.1 and 4.2).
        Condition 1: ρ = -m / sqrt(m² + σ²)
        Condition 2: a = bσ√(1 − ρ²)
        """
        m, sigma, rho, a, b = self.m, self.sigma, self.rho, self.a, self.b
        denom = math.sqrt(m ** 2 + sigma ** 2)
        cond1 = abs(rho + m / denom) < tol
        cond2 = abs(a - b * sigma * math.sqrt(1 - rho ** 2)) < tol
        return cond1 and cond2

    def non_negative_variance_constraints(self) -> dict[str, bool]:
        """
        Checks the positivity constraints from Section 4.3 (Equations 4.3a-d).
        """
        a, b, rho, sigma = self.a, self.b, self.rho, self.sigma
        return {
            "sigma >= 0":                    sigma >= 0,
            "b >= 0":                        b >= 0,
            "|rho| <= 1":                    abs(rho) <= 1,
            "a + b*sigma*sqrt(1-rho²) >= 0": (
                a + b * sigma * math.sqrt(max(0.0, 1 - rho ** 2)) >= 0
            ),
        }
