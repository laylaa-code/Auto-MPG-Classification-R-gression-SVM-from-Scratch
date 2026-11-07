from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from modeles_svm import SVCMulticlasse, SVRLineaire


DOSSIER = Path(__file__).resolve().parent
RAPPORT = []
ETIQUETTES = ["elevee", "moyenne", "faible"]


def afficher(texte=""):
    print(texte)
    RAPPORT.append(str(texte))


def charger_donnees():
    # Une copie CSV reste dans le dossier pour pouvoir executer le TP hors ligne.
    try:
        data = sns.load_dataset("mpg")
    except Exception:
        data = pd.read_csv(DOSSIER / "donnees" / "mpg.csv")
    avant = len(data)
    data = data.dropna(subset=["mpg", "horsepower"]).reset_index(drop=True)
    return data, avant - len(data)


def classes_mpg(y, seuil1, seuil2):
    # Plus la valeur mpg est elevee, plus la consommation en carburant est faible.
    return np.where(y <= seuil1, "elevee",
                    np.where(y <= seuil2, "moyenne", "faible"))


def scores_regression(y, predictions):
    return (mean_absolute_error(y, predictions),
            np.sqrt(mean_squared_error(y, predictions)),
            r2_score(y, predictions))


def dessiner_matrice(ax, matrice, titre):
    ax.imshow(matrice, cmap="Blues", vmin=0, vmax=max(1, matrice.max()))
    ax.set_title(titre)
    ax.set_xticks(range(3), ["Elevee", "Moyenne", "Faible"])
    ax.set_yticks(range(3), ["Elevee", "Moyenne", "Faible"])
    ax.set_xlabel("Classe predite")
    ax.set_ylabel("Classe reelle")
    for i in range(3):
        for j in range(3):
            couleur = "white" if matrice[i, j] > matrice.max() / 2 else "black"
            ax.text(j, i, str(matrice[i, j]), ha="center", va="center", color=couleur)


def creer_graphiques(y_train, y_test, seuil1, seuil2,
                      repartition_terciles, repartition_fixe,
                      matrices_svc, predictions_svr, supports_svr):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Auto MPG : comparaison des SVM", fontsize=17, fontweight="bold")

    # 1. La distribution de mpg explique le choix de la discretisation.
    ax = axes[0, 0]
    ax.hist(y_train, bins=19, color="#6484ab", edgecolor="white")
    ax.axvline(seuil1, color="#d07532", linestyle="--", linewidth=2,
               label=f"Seuil 1 = {seuil1:.1f}")
    ax.axvline(seuil2, color="#548c63", linestyle="--", linewidth=2,
               label=f"Seuil 2 = {seuil2:.1f}")
    ax.set_title("1. Repartition des mpg (train)")
    ax.set_xlabel("mpg (miles par gallon)")
    ax.set_ylabel("Nombre de voitures")
    ax.legend(fontsize=9)

    # 2. Comparaison d'un decoupage en terciles avec des seuils fixes.
    ax = axes[0, 1]
    positions = np.arange(3)
    ax.bar(positions - .19, repartition_terciles, .38,
           label="Terciles", color="#4778a7")
    ax.bar(positions + .19, repartition_fixe, .38,
           label="Seuils 20 / 30", color="#e7a35e")
    ax.set_xticks(positions, ["Elevee", "Moyenne", "Faible"])
    ax.set_title("2. Equilibre des classes (train)")
    ax.set_ylabel("Nombre de voitures")
    ax.legend(fontsize=9)

    # 3 et 4. Les erreurs par classe sont plus parlantes qu'une seule accuracy.
    dessiner_matrice(axes[0, 2], matrices_svc["lineaire"],
                    "3. SVC lineaire maison")
    dessiner_matrice(axes[1, 0], matrices_svc["rbf"],
                    "4. SVC RBF maison")

    # 5. Le nuage de points indique la taille des erreurs du SVR.
    ax = axes[1, 1]
    y_pred = predictions_svr
    mini = min(y_test.min(), y_pred.min()) - 1
    maxi = max(y_test.max(), y_pred.max()) + 1
    limites = np.array([mini, maxi])
    ax.scatter(y_test, y_pred, color="#4778a7", alpha=.72, s=28)
    ax.plot(limites, limites, color="#bb5e50", label="Prediction parfaite")
    ax.plot(limites, limites + 1, linestyle=":", color="#888888", linewidth=1)
    ax.plot(limites, limites - 1, linestyle=":", color="#888888", linewidth=1,
            label="Ecart de 1 mpg")
    ax.set_xlim(mini, maxi)
    ax.set_ylim(mini, maxi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mpg reel")
    ax.set_ylabel("mpg predit")
    ax.set_title("5. SVR maison (epsilon = 1)")
    ax.legend(loc="upper left", fontsize=8)

    # 6. Evolution du nombre de vecteurs de support quand epsilon augmente.
    ax = axes[1, 2]
    epsilons = [0.1, 0.5, 1.0, 2.0]
    ax.plot(epsilons, supports_svr["maison"], "o-", linewidth=2,
            color="#4778a7", label="Maison")
    ax.plot(epsilons, supports_svr["sklearn"], "s--", linewidth=1.5,
            color="#d07532", label="Scikit-learn")
    ax.set_xticks(epsilons)
    ax.set_xlabel("epsilon (mpg)")
    ax.set_ylabel("Vecteurs de support")
    ax.set_title("6. Influence de epsilon")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)

    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=2.3, w_pad=2.2)
    fig.savefig(DOSSIER / "graphiques_resultats.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def tracer_test_bruit(epsilons, rmse_propres, rmse_bruites, supports_propres, supports_bruites):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.suptitle("SVR lineaire : effet du bruit sur les donnees d'entrainement",
                 fontsize=13, fontweight="bold")
    axes[0].plot(epsilons, rmse_propres, "o-", label="Train sans bruit")
    axes[0].plot(epsilons, rmse_bruites, "s--", label="Train avec bruit")
    axes[0].set_title("Erreur sur le meme test non bruite")
    axes[0].set_ylabel("RMSE (mpg)")
    axes[0].set_xlabel("epsilon (mpg)")
    axes[0].legend()

    axes[1].plot(epsilons, supports_propres, "o-", label="Train sans bruit")
    axes[1].plot(epsilons, supports_bruites, "s--", label="Train avec bruit")
    axes[1].set_title("Vecteurs de support du modele maison")
    axes[1].set_ylabel("Nombre de vecteurs de support")
    axes[1].set_xlabel("epsilon (mpg)")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(epsilons)
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(DOSSIER / "test_bruit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    data, nb_supprimes = charger_donnees()
    afficher("DEVOIR SVM - AUTO MPG")
    afficher("=" * 63)
    afficher(f"Donnees : {len(data)} voitures ({nb_supprimes} lignes incompletes retirees).")

    numeriques = ["displacement", "horsepower", "weight", "acceleration", "model_year"]
    categorielles = ["cylinders", "origin"]
    X = data[numeriques + categorielles]  # Le nom du vehicule n'est pas utilise.
    y = data["mpg"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    pretraitement = ColumnTransformer([
        ("numeriques", StandardScaler(), numeriques),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorielles),
    ])
    X_train_p = np.asarray(pretraitement.fit_transform(X_train), dtype=float)
    X_test_p = np.asarray(pretraitement.transform(X_test), dtype=float)

    afficher(f"Separation : {len(X_train)} pour entrainement, {len(X_test)} pour test.")
    afficher("Pretraitement : standardisation des 5 variables numeriques et")
    afficher("encodage one-hot des cylindres et de l'origine ; nom exclu.")
    afficher("Les transformations sont ajustees uniquement sur l'entrainement.")

    afficher("\n1. CLASSIFICATION SVC")
    afficher("-" * 63)
    seuil1, seuil2 = np.quantile(y_train, [1/3, 2/3])
    yc_train = classes_mpg(y_train, seuil1, seuil2)
    yc_test = classes_mpg(y_test, seuil1, seuil2)
    afficher(f"Seuils des terciles calcules sur train : {seuil1:.2f} et {seuil2:.2f} mpg.")
    afficher("Consommation elevee : mpg <= seuil 1 ; moyenne : entre les deux ;")
    afficher("consommation faible : mpg > seuil 2 (mpg eleve = consommation faible).")
    afficher("Repartition (train / test) :")
    repartition_terciles = []
    for etiquette in ETIQUETTES:
        n_train = int(np.sum(yc_train == etiquette))
        n_test = int(np.sum(yc_test == etiquette))
        repartition_terciles.append(n_train)
        afficher(f"  {etiquette:8s} : {n_train:3d} / {n_test:2d}")

    # Un second choix de seuils sert seulement a discuter le desequilibre.
    classes_fixes = classes_mpg(y_train, 20, 30)
    repartition_fixe = [int(np.sum(classes_fixes == e)) for e in ETIQUETTES]
    afficher("Comparaison avec des seuils fixes (20 et 30 mpg), sur train :")
    afficher("  Terciles : " + " / ".join(map(str, repartition_terciles)))
    afficher("  Fixes    : " + " / ".join(map(str, repartition_fixe)))
    afficher("Les terciles equilibrent mieux les groupes ; les seuils fixes")
    afficher("sont plus simples a interpreter mais moins equilibres ici.")
    afficher("Les voitures proches des seuils risquent d'etre mal classees.")

    afficher("\nComparaison sur le meme jeu de test (C = 2) :")
    afficher(f"{'Modele':23s} {'Accuracy':>10s} {'F1 macro':>10s}")
    matrices_svc = {}
    pred_lineaire = None
    for noyau in ("lineaire", "rbf"):
        gamma = 1.0 / X_train_p.shape[1]
        maison = SVCMulticlasse(C=2.0, noyau=noyau, gamma=gamma)
        maison.fit(X_train_p, yc_train)
        pred_maison = maison.predict(X_test_p)
        matrices_svc[noyau] = confusion_matrix(yc_test, pred_maison, labels=ETIQUETTES)
        if noyau == "lineaire":
            pred_lineaire = pred_maison

        reference = SVC(C=2.0, kernel="linear" if noyau == "lineaire" else "rbf",
                        gamma=gamma, decision_function_shape="ovo")
        reference.fit(X_train_p, yc_train)
        pred_reference = reference.predict(X_test_p)
        for titre, pred in ((f"SVC {noyau} maison", pred_maison),
                            (f"SVC {noyau} sklearn", pred_reference)):
            afficher(f"{titre:23s} {accuracy_score(yc_test, pred):10.3f} "
                    f"{f1_score(yc_test, pred, average='macro'):10.3f}")
            afficher("Matrice de confusion, ordre elevee / moyenne / faible :")
            afficher(np.array2string(confusion_matrix(yc_test, pred, labels=ETIQUETTES)))
        afficher("")

    # Mesurer si les erreurs se concentrent autour des deux seuils.
    distance_seuil = np.minimum(abs(y_test - seuil1), abs(y_test - seuil2))
    proches = distance_seuil <= 2
    erreurs_proches = int(np.sum(pred_lineaire[proches] != yc_test[proches]))
    erreurs_loin = int(np.sum(pred_lineaire[~proches] != yc_test[~proches]))
    afficher("Analyse des erreurs du SVC lineaire (test) :")
    afficher(f"  A 2 mpg ou moins d'un seuil : {erreurs_proches} erreurs sur {sum(proches)} voitures.")
    afficher(f"  A plus de 2 mpg des seuils : {erreurs_loin} erreurs sur {sum(~proches)} voitures.")
    afficher("Les erreurs se concentrent pres des frontieres de classes.")
    afficher("Sur ce test, le modele lineaire fait mieux que le RBF ;")
    afficher("un noyau non lineaire n'ameliore donc pas forcement le resultat.")

    afficher("\n2. REGRESSION SVR LINEAIRE")
    afficher("-" * 63)
    afficher("Perte epsilon-insensible : max(0, |mpg - prediction| - epsilon).")
    afficher("Les erreurs a l'interieur du tube +/- epsilon ne sont pas penalisees.")
    afficher("C = 1 pour les deux implementations ; meme train et meme test.")
    afficher(f"{'epsilon':>8s} {'modele':>10s} {'MAE':>8s} {'RMSE':>8s} "
            f"{'R2':>8s} {'support':>9s} {'dans tube':>10s}")

    supports_svr = {"maison": [], "sklearn": []}
    rmse_maison = []
    predictions_svr = None
    for epsilon in (0.1, 0.5, 1.0, 2.0):
        maison = SVRLineaire(C=1.0, epsilon=epsilon)
        maison.fit(X_train_p, y_train)
        reference = SVR(C=1.0, epsilon=epsilon, kernel="linear", tol=1e-5,
                        max_iter=100000)
        reference.fit(X_train_p, y_train)
        for titre, modele, nb_support in (("maison", maison, maison.n_support_),
                                         ("sklearn", reference, len(reference.support_))):
            predictions = modele.predict(X_test_p)
            if epsilon == 1.0 and titre == "maison":
                predictions_svr = predictions
            mae, rmse, r2 = scores_regression(y_test, predictions)
            if titre == "maison":
                rmse_maison.append(rmse)
            dans_tube = np.sum(np.abs(y_train - modele.predict(X_train_p)) <= epsilon)
            supports_svr[titre].append(nb_support)
            afficher(f"{epsilon:8.1f} {titre:>10s} {mae:8.3f} {rmse:8.3f} "
                    f"{r2:8.3f} {nb_support:9d} {dans_tube:10d}")

    afficher("\nInterpretation : quand epsilon augmente, le modele ignore")
    afficher("davantage de petits ecarts et utilise moins de vecteurs de support.")
    afficher("En contrepartie, un epsilon trop grand peut reduire la precision.")
    afficher("Le comptage maison emploie un seuil de 0.01 sur les poids duaux,")
    afficher("ce qui peut creer de faibles differences avec scikit-learn.")
    afficher("Les deux implementations donnent ici des performances tres proches.")

    # Ajouter une petite erreur de mesure aux mpg de l'entrainement uniquement.
    # On garde exactement le meme jeu de test, sans bruit, pour comparer.
    bruit = np.random.default_rng(42).normal(0, 2.0, len(y_train))
    y_train_bruite = y_train + bruit
    epsilons = [0.1, 0.5, 1.0, 2.0]
    rmse_bruites, supports_bruites = [], []
    afficher("\nTest avec bruit : ajout d'un bruit gaussien d'ecart-type 2 mpg")
    afficher("sur la cible d'entrainement seulement ; test laisse intact.")
    afficher(f"{'epsilon':>8s} {'RMSE normal':>13s} {'RMSE bruit':>12s} {'support bruit':>14s}")
    for epsilon, rmse_normal in zip(epsilons, rmse_maison):
        modele_bruite = SVRLineaire(C=1.0, epsilon=epsilon)
        modele_bruite.fit(X_train_p, y_train_bruite)
        _, rmse_bruit, _ = scores_regression(y_test, modele_bruite.predict(X_test_p))
        rmse_bruites.append(rmse_bruit)
        supports_bruites.append(modele_bruite.n_support_)
        afficher(f"{epsilon:8.1f} {rmse_normal:13.3f} {rmse_bruit:12.3f} "
                f"{modele_bruite.n_support_:14d}")
    afficher("Avec du bruit, l'erreur sur le test augmente legerement ici.")
    afficher("Un epsilon plus grand ignore davantage de petits ecarts et reduit")
    afficher("les vecteurs de support, mais n'assure pas un meilleur RMSE.")
    tracer_test_bruit(epsilons, rmse_maison, rmse_bruites,
                     supports_svr["maison"], supports_bruites)

    creer_graphiques(y_train, y_test, seuil1, seuil2,
                     repartition_terciles, repartition_fixe,
                     matrices_svc, predictions_svr, supports_svr)
    (DOSSIER / "resultats.txt").write_text("\n".join(RAPPORT) + "\n", encoding="utf-8")
    afficher("\nResultats enregistres dans resultats.txt")
    afficher("Graphiques enregistres dans graphiques_resultats.png et test_bruit.png")


if __name__ == "__main__":
    main()
