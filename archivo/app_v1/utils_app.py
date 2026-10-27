"""
app/utils_app.py
================
Funciones de utilidad, inferencia y lógica de negocio para la aplicación web
de Riesgo de Crédito y Pérdida Esperada (Lending Club).

Universidad Nacional de Colombia - Facultad de Minas
"""

import numpy as np

class ScoringEngine:
    def __init__(self, score_base=600.0, odds_base=50.0, pdo=20.0):
        self.score_base = float(score_base)
        self.odds_base = float(odds_base)
        self.pdo = float(pdo)
        self.factor = self.pdo / np.log(2.0)
        self.offset = self.score_base - self.factor * np.log(self.odds_base)

    def calcular_pd(self, int_rate, dti, annual_inc, term_months, emp_years, home_ownership):
        """
        Estima la Probabilidad de Incumplimiento (PD) a partir del perfil socioeconómico.
        Parametrizado a partir de los coeficientes de la red y la regresión logística.
        """
        # Efecto de variables clave (coeficientes estandarizados)
        logit = -3.15
        logit += (float(int_rate) - 13.0) * 0.115       # Tasa de interés (+)
        logit += (float(dti) - 17.0) * 0.042            # DTI (+)
        logit -= (float(annual_inc) - 65000) / 50000.0 * 0.28  # Ingresos (-)
        
        if int(term_months) == 60:
            logit += 0.36   # Plazo 60 meses (+)
        
        if home_ownership == "RENT":
            logit += 0.12
        elif home_ownership == "OWN":
            logit -= 0.08
            
        logit -= min(float(emp_years), 10) * 0.015      # Antigüedad laboral (-)

        pd_val = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(pd_val, 0.008, 0.96))

    def pd_a_score(self, pd_val):
        """Transforma PD a Scorecard usando Factor y Offset formales."""
        p_safe = np.clip(pd_val, 1e-6, 1.0 - 1e-6)
        raw_score = self.offset - self.factor * np.log(p_safe / (1.0 - p_safe))
        return int(np.clip(np.round(raw_score), 300, 850))

    def calcular_lgd(self, int_rate, annual_inc, term_months):
        """
        Estima la Pérdida Dado el Incumplimiento (LGD).
        En consonancia con docs/parte2_lgd.md:
        Media observada: 0.930. Clientes de 60 meses amortizan más despacio -> mayor LGD.
        """
        lgd_base = 0.925
        lgd_base += (float(int_rate) - 13.0) * 0.0025
        lgd_base -= (float(annual_inc) - 65000) / 100000.0 * 0.004
        if int(term_months) == 60:
            lgd_base += 0.015
        return float(np.clip(lgd_base, 0.20, 0.99))

    def calcular_ead(self, loan_amnt, term_months):
        """
        Exposición al Incumplimiento (EAD) al momento de la originación (Sección 8 de LGD):
        f_bar = 0.670 para 36 meses, 0.846 para 60 meses.
        """
        factor_amortizacion = 0.846 if int(term_months) == 60 else 0.670
        return float(loan_amnt) * factor_amortizacion

    def calcular_perdida_esperada(self, pd_val, lgd_val, ead_val):
        """E[L] = PD * LGD * EAD"""
        return float(pd_val * lgd_val * ead_val)

    def calcular_percentil(self, score):
        """
        Calcula el percentil poblacional aproximado sobre la distribución histórica
        de Lending Club (Media: 625, Desv: 68).
        """
        from scipy.stats import norm
        z = (score - 625.0) / 68.0
        pct = norm.cdf(z) * 100.0
        return int(np.clip(np.round(pct), 1, 99))
