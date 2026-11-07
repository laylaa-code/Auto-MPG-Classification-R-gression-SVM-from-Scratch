import numpy as np


class SVCBinaire:
    """SVM binaire appris avec l'algorithme SMO."""

    def __init__(self, C=2.0, noyau="lineaire", gamma=0.1,
                 tolerance=1e-3, max_passes=20, max_iterations=30000,
                 graine=42):
        self.C = C
        self.noyau = noyau
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_passes = max_passes
        self.max_iterations = max_iterations
        self.graine = graine

    def _noyau(self, X, Z):
        if self.noyau == "lineaire":
            return X @ Z.T
        if self.noyau == "rbf":
            distances = (np.sum(X**2, axis=1)[:, None]
                         + np.sum(Z**2, axis=1)[None, :]
                         - 2 * X @ Z.T)
            return np.exp(-self.gamma * np.maximum(distances, 0))
        raise ValueError("Noyau inconnu")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        K = self._noyau(X, X)
        alphas = np.zeros(n)
        b = 0.0
        erreurs = -y.copy()  # f(x) - y au depart
        rng = np.random.default_rng(self.graine)
        passes = 0
        iterations = 0

        while passes < self.max_passes and iterations < self.max_iterations:
            changements = 0
            for i in rng.permutation(n):
                Ei = erreurs[i]
                if not ((y[i] * Ei < -self.tolerance and alphas[i] < self.C - 1e-9)
                        or (y[i] * Ei > self.tolerance and alphas[i] > 1e-9)):
                    continue

                candidats = np.flatnonzero((alphas > 1e-8) & (alphas < self.C - 1e-8))
                candidats = candidats[candidats != i]
                if len(candidats):
                    j = candidats[np.argmax(np.abs(Ei - erreurs[candidats]))]
                else:
                    j = rng.integers(n - 1)
                    if j >= i:
                        j += 1
                Ej = erreurs[j]
                ai, aj = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L = max(0.0, aj - ai)
                    H = min(self.C, self.C + aj - ai)
                else:
                    L = max(0.0, ai + aj - self.C)
                    H = min(self.C, ai + aj)
                if H - L < 1e-12:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= -1e-12:
                    continue
                nouveau_j = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)
                if abs(nouveau_j - aj) < 1e-7:
                    continue
                nouveau_i = ai + y[i] * y[j] * (aj - nouveau_j)

                b1 = b - Ei - y[i] * (nouveau_i - ai) * K[i, i] - y[j] * (nouveau_j - aj) * K[i, j]
                b2 = b - Ej - y[i] * (nouveau_i - ai) * K[i, j] - y[j] * (nouveau_j - aj) * K[j, j]
                if 1e-8 < nouveau_i < self.C - 1e-8:
                    nouveau_b = b1
                elif 1e-8 < nouveau_j < self.C - 1e-8:
                    nouveau_b = b2
                else:
                    nouveau_b = (b1 + b2) / 2

                erreurs += (y[i] * (nouveau_i - ai) * K[:, i]
                            + y[j] * (nouveau_j - aj) * K[:, j]
                            + nouveau_b - b)
                b = nouveau_b
                alphas[i] = nouveau_i
                alphas[j] = nouveau_j
                changements += 1

            iterations += 1
            if changements == 0:
                passes += 1
            else:
                passes = 0

        supports = alphas > 1e-7
        self.X_support_ = X[supports]
        self.coefficients_ = alphas[supports] * y[supports]
        self.b_ = b
        self.n_support_ = int(np.sum(supports))
        self.iterations_ = iterations
        return self

    def decision_function(self, X):
        return self._noyau(np.asarray(X, dtype=float), self.X_support_) @ self.coefficients_ + self.b_


class SVCMulticlasse:
    """Classification en trois classes avec un SVM binaire par paire de classes."""

    def __init__(self, C=2.0, noyau="lineaire", gamma=0.1):
        self.C = C
        self.noyau = noyau
        self.gamma = gamma

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.modeles_ = []
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                masque = (y == self.classes_[i]) | (y == self.classes_[j])
                y_binaire = np.where(y[masque] == self.classes_[j], 1, -1)
                modele = SVCBinaire(C=self.C, noyau=self.noyau, gamma=self.gamma)
                modele.fit(X[masque], y_binaire)
                self.modeles_.append((i, j, modele))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)))
        scores = np.zeros_like(votes)
        for i, j, modele in self.modeles_:
            distances = modele.decision_function(X)
            votes[:, i] += distances < 0
            votes[:, j] += distances >= 0
            scores[:, i] -= distances
            scores[:, j] += distances
        # Les scores ne servent qu'a departager les egalites de votes.
        scores /= np.maximum(1.0, np.max(np.abs(scores), axis=1, keepdims=True))
        return self.classes_[np.argmax(votes + scores * 0.01, axis=1)]

    @property
    def n_support_(self):
        return sum(modele.n_support_ for _, _, modele in self.modeles_)


class SVRLineaire:
    """SVR lineaire : 0.5*||w||² + C*somme(max(0, |y-f(x)|-epsilon)).

    Resolution par ADMM : aucune classe SVM de scikit-learn n'est utilisee.
    """

    def __init__(self, C=1.0, epsilon=0.5, rho=1.0,
                 max_iterations=15000, tolerance=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.rho = rho
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        A = np.column_stack([X, np.ones(n)])
        regularisation = np.diag(np.r_[np.ones(p), 0.0])
        matrice = regularisation + self.rho * (A.T @ A)
        theta = np.zeros(p + 1)
        r = np.zeros(n)
        u = np.zeros(n)

        for iteration in range(self.max_iterations):
            theta = np.linalg.solve(matrice, self.rho * A.T @ (y + r - u))
            v = A @ theta - y + u
            seuil = self.C / self.rho
            absolu = np.abs(v)
            r_ancien = r.copy()
            r = np.where(
                absolu <= self.epsilon, v,
                np.where(absolu <= self.epsilon + seuil,
                         np.sign(v) * self.epsilon,
                         v - np.sign(v) * seuil)
            )
            residu = A @ theta - y - r
            u += residu

            primal = np.linalg.norm(residu)
            dual = self.rho * np.linalg.norm(A.T @ (r - r_ancien))
            limite_p = np.sqrt(n) * self.tolerance + self.tolerance * max(
                np.linalg.norm(A @ theta - y), np.linalg.norm(r))
            limite_d = np.sqrt(p + 1) * self.tolerance + self.tolerance * np.linalg.norm(
                self.rho * A.T @ u)
            if primal <= limite_p and dual <= limite_d:
                break

        self.coef_ = theta[:p]
        self.intercept_ = theta[-1]
        self.iterations_ = iteration + 1
        # Le multiplicateur de la contrainte ADMM correspond aux poids duaux.
        self.n_support_ = int(np.sum(np.abs(self.rho * u) > 0.01))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
