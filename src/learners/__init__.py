from .cola_learner import COLALearner
from .mae_nq_learner import MAENQLearner
from .owm_nq_learner import OWM_NQLearner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner import NQLearner
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .fmac_learner import FMACLearner
from .rvae_damaq_qatten_learner import RVAEDMAQ_qattenLearner
from .rvae_fmac_learner import RVAEFMACLearner
from .rvae_nq_learner import RVAENQLearner
from .state_attention_nq_learner import SAttNQLearner
from .state_rvae_nq_learner import SRVAENQLearner
from .vae_nq_learner import VAENQLearner
from .rvae_q_learner import RVaeQLearner
from .world_nq_learner import WORLDNQLearner
from .world_nq_learner_updated import WORLD_NQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["world_nq_learner"] = WORLDNQLearner
REGISTRY["world_nq_learner_updated"] = WORLD_NQLearner
REGISTRY["owm_nq_learner"] = OWM_NQLearner
REGISTRY["rvae_q_learner"] = RVaeQLearner
REGISTRY["rvae_nq_learner"]= RVAENQLearner
REGISTRY["srvae_nq_learner"]= SRVAENQLearner
REGISTRY["satt_nq_learner"]= SAttNQLearner
REGISTRY['mae_nq_learner'] = MAENQLearner
REGISTRY["vae_nq_learner"] = VAENQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["rvae_dmaq_qatten_learner"] = RVAEDMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["rvae_fmac_learner"] = RVAEFMACLearner
REGISTRY["cola_learner"] = COLALearner
