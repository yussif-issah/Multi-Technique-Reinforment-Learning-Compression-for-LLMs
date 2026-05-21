# Environment.py — fixed
class Environment:
    def __init__(self, num_actions, num_states, start, terminal):
        self.num_actions  = num_actions
        self.num_states   = num_states
        self.start        = start
        self.terminal     = terminal
        self.layers       = [{'importance': 0.027898322322082142, 'sparsity_ratio': 0.07523850455802956}, {'importance': 0.03841185675271865, 'sparsity_ratio': 0.060006136721279954}, {'importance': 0.040090922448029156, 'sparsity_ratio': 0.04693264541230671}, {'importance': 0.03985843642867847, 'sparsity_ratio': 0.04699794986705088}, {'importance': 0.040426735587091256, 'sparsity_ratio': 0.0457305957616302}, {'importance': 0.037481912675315913, 'sparsity_ratio': 0.04541573005636739}, {'importance': 0.03430460374418989, 'sparsity_ratio': 0.04660691374941811}, {'importance': 0.032573874489023685, 'sparsity_ratio': 0.04660163642211282}, {'importance': 0.029732378696959764, 'sparsity_ratio': 0.04590709468861318}, {'importance': 0.029913201156454742, 'sparsity_ratio': 0.04547128529128633}, {'importance': 0.029034920638907714, 'sparsity_ratio': 0.045519389019111275}, {'importance': 0.028440789700567074, 'sparsity_ratio': 0.04567507017461747}, {'importance': 0.027717499862587168, 'sparsity_ratio': 0.04504755741573986}, {'importance': 0.02774333164251502, 'sparsity_ratio': 0.04465121432289559}, {'importance': 0.028130808341432823, 'sparsity_ratio': 0.04474294988602554}, {'importance': 0.028518285040350637, 'sparsity_ratio': 0.044236064574879067}, {'importance': 0.025508882678755663, 'sparsity_ratio': 0.04386192282246802}, {'importance': 0.025263480769441054, 'sparsity_ratio': 0.043665213906085555}, {'importance': 0.026090097727132375, 'sparsity_ratio': 0.043308297587182236}, {'importance': 0.025793032257962055, 'sparsity_ratio': 0.04333174908099397}, {'importance': 0.026968378244679406, 'sparsity_ratio': 0.04310598521652617}, {'importance': 0.028828266399484884, 'sparsity_ratio': 0.043092648600049584}, {'importance': 0.029990696496238305, 'sparsity_ratio': 0.042775440710196226}, {'importance': 0.03068815455429036, 'sparsity_ratio': 0.042242420769726056}, {'importance': 0.03148893973205383, 'sparsity_ratio': 0.04278859943923555}, {'importance': 0.0320055753306109, 'sparsity_ratio': 0.04200166494735164}, {'importance': 0.033555482126282135, 'sparsity_ratio': 0.042124585166496316}, {'importance': 0.03445959442375702, 'sparsity_ratio': 0.04128729370591554}, {'importance': 0.9684850930550961, 'sparsity_ratio': 0.04142998413718426}, {'importance': 0.9251651981160851, 'sparsity_ratio': 0.04165298086373917}, {'importance': 0.8865466871239437, 'sparsity_ratio': 0.04118906525132569}, {'importance': 1.0242042423594766, 'sparsity_ratio': 0.04129410284170833}]

        self.action_aggressiveness = {
            0: 0.00,   # KEEP
            1: 0.01,   # INT8
            2: 0.05,   # INT4
            3: 0.25,   # PRUNE15
            4: 0.55,   # PRUNE25
            5: 0.75,   # PRUNE50
        }
        self.action_bits = {
            0: 16,     # KEEP
            1: 8,      # INT8
            2: 4,      # INT4
            3: 2,      # INT2
            4: 0,      # PRUNE15
            5: 0,      # PRUNE25
            6: 0,      # PRUNE50
        }

    def dynamics(self, state, action):
        if state == self.terminal:
            return [state], [0.0], [True], [False], [1.0]

        next_state = state + 1

        # compression gain
        bits_before     = 16.0
        bits_after      = self.action_bits[action]
        compression_gain = (bits_before - bits_after) / bits_before

        # ppl hit
        ppl_hit = self.PPL_hit(state, action)

        terminated = (next_state == self.terminal)
        truncated  = False
        prob       = 1.0

        # reward returned as ppl_hit only — compression gain added in TrainAgent
        # OR compute full reward here — pick one place, not both
        return [next_state], [ppl_hit], [terminated], [truncated], [prob]

    def PPL_hit(self, state, action):
        layer = self.layers[state]
        agg   = self.action_aggressiveness[action]
        return agg * layer['importance'] * (1 - layer['sparsity_ratio']/2)