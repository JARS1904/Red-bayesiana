#Use este en el documento
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red
model = BayesianNetwork([('Fiebre', 'Diagnóstico'), ('Congestión Nasal', 'Diagnóstico')])

# Definir las distribuciones de probabilidad condicional (CDPs)
cdp_fiebre = TabularCPD(variable='Fiebre', variable_card=2, values=[[0.4], [0.6]])
cdp_congestion = TabularCPD(variable='Congestión Nasal', variable_card=2, values=[[0.7], [0.3]])

cdp_diagnostico = TabularCPD(variable='Diagnóstico', variable_card=2,
                            values=[[0.8, 0.2, 0.9, 0.1], [0.2, 0.8, 0.1, 0.9]],
                            evidence=['Fiebre', 'Congestión Nasal'],
                            evidence_card=[2, 2])

# Asignar las CDPs al modelo
model.add_cpds(cdp_fiebre, cdp_congestion, cdp_diagnostico)

# Verificar si la red bayesiana es válida
if model.check_model() == True:
    print("El modelo es valido\n")

# Hacer inferencia en la red bayesiana
inference = VariableElimination(model)

# Calcular la probabilidad de tener gripe dada una fiebre alta y congestión nasal presente
print("Probabilidad de tener gripe dada una fiebre alta y congestión nasal presente")
prob_gripe = inference.query(variables=['Diagnóstico'], evidence={'Fiebre': 1, 'Congestión Nasal': 1})
print(prob_gripe, "\n")

# Calcular la probabilidad de tener resfriado dada una fiebre normal y congestión nasal ausente
print("Probabilidad de tener resfriado dada una fiebre normal y congestión nasal ausente")
prob_resfriado = inference.query(variables=['Diagnóstico'], evidence={'Fiebre': 0, 'Congestión Nasal': 0})
print(prob_resfriado)
