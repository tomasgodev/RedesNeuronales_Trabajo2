"""
scripts/scorecard_engine.py
===========================
Motor de Scorecard Crediticia Formal para Riesgo de Crédito.
Derivado matemáticamente a partir de Odds y Log-Odds según los estándares
internacionales de la industria (Siddiqi, 2012; Anderson, 2007).

Universidad Nacional de Colombia - Facultad de Minas
Curso: Redes Neuronales y Algoritmos Bioinspirados
"""

import numpy as np

class CreditScorecard:
    def __init__(self, score_base=600.0, odds_base=50.0, pdo=20.0, min_score=300, max_score=850):
        """
        Inicializa la scorecard crediticia con sus parámetros de calibración.

        Parámetros:
        -----------
        score_base : float
            Puntaje base de referencia (ej. 600 puntos).
        odds_base : float
            Odds de referencia (Buenos:Malos) para el puntaje base (ej. 50:1, lo que
            equivale a una tasa de default base p_0 = 1 / (1 + 50) ≈ 1.96%).
        pdo : float
            Points to Double the Odds (Puntos para duplicar las odds).
            Cantidad de puntos en que aumenta el score cuando las odds de buen pagador se duplican.
        min_score : int
            Límite inferior operativo (estándar FICO: 300).
        max_score : int
            Límite superior operativo (estándar FICO: 850).
        """
        self.score_base = float(score_base)
        self.odds_base = float(odds_base)
        self.pdo = float(pdo)
        self.min_score = int(min_score)
        self.max_score = int(max_score)

        # Derivación matemática unívoca
        self.factor = self.pdo / np.log(2.0)
        self.offset = self.score_base - self.factor * np.log(self.odds_base)

    def probabilidad_a_odds(self, p):
        """
        Calcula las odds de buen crédito a partir de la probabilidad de incumplimiento p:
        Odds = (1 - p) / p
        """
        p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
        return (1.0 - p_safe) / p_safe

    def probabilidad_a_score(self, p):
        """
        Transforma una probabilidad de incumplimiento p (o vector de probabilidades)
        en puntaje de scorecard:
        Score = Offset + Factor * ln((1 - p) / p) = Offset - Factor * logit(p)
        """
        p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
        logit_p = np.log(p_safe / (1.0 - p_safe))
        raw_score = self.offset - self.factor * logit_p
        clipped_score = np.clip(raw_score, self.min_score, self.max_score)
        if np.isscalar(clipped_score):
            return int(np.round(clipped_score))
        return np.round(clipped_score).astype(int)

    def score_a_probabilidad(self, score):
        """
        Transformación inversa: recupera la probabilidad de default p a partir del score:
        p = 1 / (1 + exp((Score - Offset) / Factor))
        """
        odds = np.exp((score - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)

    def obtener_resumen_calibracion(self):
        """Retorna un diccionario con los parámetros formales de la scorecard."""
        return {
            "score_base": self.score_base,
            "odds_base": self.odds_base,
            "pdo": self.pdo,
            "factor": round(self.factor, 4),
            "offset": round(self.offset, 4),
            "formula_latex": (
                f"Score = {self.offset:.2f} - {self.factor:.2f} \\cdot \\ln\\left(\\frac{{p}}{{1-p}}\\right)"
            ),
            "escala_operativa": f"[{self.min_score}, {self.max_score}]"
        }

if __name__ == "__main__":
    sc = CreditScorecard(score_base=600, odds_base=50, pdo=20)
    resumen = sc.obtener_resumen_calibracion()
    print("=" * 60)
    print("RESUMEN DE CALIBRACIÓN DE SCORECARD")
    print("=" * 60)
    for k, v in resumen.items():
        print(f"  {k:18s}: {v}")
    
    print("\nEjemplos de Calibración:")
    test_probs = [0.005, 0.0196, 0.05, 0.10, 0.233, 0.50, 0.80]
    for p in test_probs:
        s = sc.probabilidad_a_score(p)
        print(f"  PD = {p*100:5.2f}%  -->  Score = {s:3d} pts")
    print("=" * 60)

r"""
Posible Codigo:

python scripts/scorecard_engine.py

============================================================
RESUMEN DE CALIBRACIÓN DE SCORECARD
============================================================
  score_base        : 600.0
  odds_base         : 50.0
  pdo               : 20.0
  factor            : 28.8539
  offset            : 487.1229
  formula_latex     : Score = 487.12 - 28.85 \cdot \ln\left(\frac{p}{1-p}\right)
  escala_operativa  : [300, 850]

Ejemplos de Calibración:
  PD =  0.50%  -->  Score = 640 pts
  PD =  1.96%  -->  Score = 600 pts
  PD =  5.00%  -->  Score = 572 pts
  PD = 10.00%  -->  Score = 551 pts
  PD = 23.30%  -->  Score = 522 pts
  PD = 50.00%  -->  Score = 487 pts
  PD = 80.00%  -->  Score = 447 pts
============================================================
"""