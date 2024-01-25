import numpy as np
from architecture_utils.preprocessing_unit_lib_ import unit_lib
from sklearn.utils.extmath import _safe_accumulator_op
from sklearn.utils.validation import _assert_all_finite


class PreprocessingScheme(unit_lib):
    def __init__(self, args, actions_index, len_std_per_modality):
        super(PreprocessingScheme, self).__init__(args)

        # 1. Define properties of the computation graph.
        self.moi = False

        # 2. Parse the preprocessing schemes.
        if actions_index[0] != None:
            self.doubly_linked_list = self.parse_actions_index(actions_index, len_std_per_modality)
        else:
            self.doubly_linked_list = actions_index[1]

        self.operation_path = []
        self.arity_list = []

    def parse_actions_index(self, actions_index, len_std_per_modality):
        # 1. Split the original index of actions into four parts: modality_A_action, modality_B_action, moi_action, dr_action.
        # 1.1 Extract modality_A_action.
        modality_X_action = []
        for i in range(len_std_per_modality[0]):
            item = self.nodes_std[actions_index[i]]
            if item != 'stop':
                modality_X_action.append(item)
        modality_X_action.reverse()

        # 1.2 Extract modality_B_action.
        modality_Y_action = []
        if self.moi == True:
            for i in range(len_std_per_modality[0], len_std_per_modality[0] + len_std_per_modality[1]):
                item = self.nodes_std[actions_index[i]]
                if item != 'stop':
                    modality_Y_action.append(item)
            modality_Y_action.reverse()

        # 1.3 Extract moi_action and dr_action.
        # dr_action = [self.nodes_dr[actions_index[-1]]]
        if self.moi == True:
            moi_action = [self.nodes_moi[actions_index[-2]]]
        else:
            moi_action = []

        # 2. Construct the doubly linked list.
        # len_dll = len(modality_X_action) + len(modality_Y_action) + len(moi_action) + len(dr_action)
        len_dll = len(modality_X_action) + len(modality_Y_action) + len(moi_action)
        if self.moi == True:
            len_dll = len_dll + 1 + 2
        else:
            len_dll = len_dll + 1 + 1
        element = np.array([['output', 1, None]])
        doubly_linked_list = np.repeat(element, len_dll, axis=0)

        # 3. Impute the doubly linked list.
        if self.moi == True:
            # 3.1 Fill the bottom blocks of the doubly linked list.
            doubly_linked_list[-2, :] = ['X', None, None]
            doubly_linked_list[-1, :] = ['Y', None, None]

            # 3.2 Fill the top blocks and middle blocks of the doubly linked list.
            # 3.2.1 Fill the top blocks of the doubly linked list.
            doubly_linked_list[1, :] = [dr_action[0], 2, None]
            if len(modality_X_action) == 0 and len(modality_Y_action) == 0:
                doubly_linked_list[2, :] = [moi_action[0], -2, -1]
            elif len(modality_X_action) == 0:
                doubly_linked_list[2, :] = [moi_action[0], -2, 3 + len(modality_X_action)]
            elif len(modality_Y_action) == 0:
                doubly_linked_list[2, :] = [moi_action[0], 3, -1]
            else:
                doubly_linked_list[2, :] = [moi_action[0], 3, 3 + len(modality_X_action)]

            # 3.2.2 Fill the middle blocks of the doubly linked list for X.
            for i in range(len(modality_X_action)):
                if i != len(modality_X_action) - 1:
                    doubly_linked_list[3 + i, :] = [modality_X_action[i], 3 + i + 1, None]
                else:
                    doubly_linked_list[3 + i, :] = [modality_X_action[i], -2, None]


            # 3.2.3 Fill the middle blocks of the doubly linked list for Y.
            for i in range(len(modality_Y_action)):
                if i != len(modality_Y_action) - 1:
                    doubly_linked_list[3 + len(modality_X_action) + i, :] = [modality_Y_action[i], 3 + len(modality_X_action) + i + 1, None]
                else:
                    doubly_linked_list[3 + len(modality_X_action) + i, :] = [modality_Y_action[i], -1, None]

        else:
            # 3.1 Fill the bottom blocks of the doubly linked list.
            doubly_linked_list[-1, :] = ['X', None, None]

            # 3.2 Fill the top blocks and middle blocks of the doubly linked list.
            # 3.2.1 Fill the top blocks of the doubly linked list.
            # doubly_linked_list[1, :] = [dr_action[0], 2, None]

            # 3.2.2 Fill the middle blocks of the doubly linked list for X.
            for i in range(len(modality_X_action)):
                if i != len(modality_X_action) - 1:
                    doubly_linked_list[1 + i, :] = [modality_X_action[i], 1 + i + 1, None]
                else:
                    doubly_linked_list[1 + i, :] = [modality_X_action[i], -1, None]
            '''for i in range(len(modality_X_action)):
                if i != len(modality_X_action) - 1:
                    doubly_linked_list[2 + i, :] = [modality_X_action[i], 2 + i + 1, None]
                else:
                    doubly_linked_list[2 + i, :] = [modality_X_action[i], -1, None]'''

        return doubly_linked_list


    def post_order(self, time = 0):
        #Traversal the tree to obtain the operation path.

        #1. Stop Criterion.
        if time == None:
            return
        elif (self.doubly_linked_list[time, 1] == None) and (self.doubly_linked_list[time, 2] == None):
            #Store the information of the input nodes.
            self.operation_path.append(self.doubly_linked_list[time, 0])
            self.arity_list.append('input')
            return

        #2. Post-Order Traversal.
        self.post_order(time = self.doubly_linked_list[time, 1])
        self.post_order(time = self.doubly_linked_list[time, 2])

        #3. Store the information of the operation or output nodes.
        self.operation_path.append(self.doubly_linked_list[time, 0])
        if self.doubly_linked_list[time, 0] in self.node_multi_omics_integration:
            self.arity_list.append(2)
        elif self.doubly_linked_list[time, 0] in (self.node_dimensionality_reduction + self.node_feature_selection + self.node_scaling + self.node_normalization + self.node_stop):
            self.arity_list.append(1)
        elif self.doubly_linked_list[time, 0] in ['output']:
            self.arity_list.append('output')

    def nan_inf_check(self, x):
        if isinstance(x, np.ndarray):
            if np.isnan(x).sum() > 0 or np.isinf(x).sum() > 0 or (not np.isfinite(_safe_accumulator_op(np.sum, x))):
                return True
            try:
                _assert_all_finite(x)
            except:
                return True
        else:
            if x == None or np.isnan(x) or np.isinf(x):
                return True
            else:
                return False

    def compute(self, x, x_gene_id, y = None, y_gene_id = None):
        #Compute the processed data according to the computation graph.

        #1. Get the operation order via post-order-traversal.
        self.post_order()

        #2. Execute the operation via stack-structured-buffer-area.
        buffer_data = []
        buffer_gene_id = []
        for i in range(len(self.operation_path)):
            if self.arity_list[i] == 'input': #Add the input data into buffer area.
                if self.operation_path[i] == 'X':
                    buffer_data.append(x)
                    buffer_gene_id.append(x_gene_id)
                elif self.operation_path[i] == 'Y':
                    buffer_data.append(y)
                    buffer_gene_id.append(y_gene_id)
                else:
                    print('INPUT NODE IS OUT OF EXPECTION: ' + self.operation_path[i])
            elif self.arity_list[i] == 1:  #Execute the one-arity-operation and update the buffer areas.
                if self.nan_inf_check(buffer_data[-1]):
                    return None, None
                else:
                    print(self.operation_path[i])
                    result = self.get_operation(opt_name = self.operation_path[i], x = buffer_data[-1]
                                            , x_gene_id = buffer_gene_id[-1], y = None, y_gene_id = None)
                if not isinstance(result, tuple):
                    buffer_data.pop()
                    buffer_data.append(result)
                elif isinstance(result, tuple):
                    buffer_data.pop()
                    buffer_data.append(result[0])
                    buffer_gene_id.pop()
                    buffer_gene_id.append(result[1])
                else:
                    print('OPERATION ERROR')
            elif self.arity_list[i] == 2:  #Execute the two-arity-operation and update the buffer areas.
                if self.nan_inf_check(buffer_data[-1]):
                    return None, None
                elif self.nan_inf_check(buffer_data[-2]):
                    return None, None
                else:
                    print(self.operation_path[i])
                    result = self.get_operation(opt_name=self.operation_path[i], x=buffer_data[-2],
                                            x_gene_id=buffer_gene_id[-2], y=buffer_data[-1], y_gene_id = buffer_gene_id[-1])
                buffer_data.pop()
                buffer_data.pop()
                buffer_gene_id.pop()
                buffer_gene_id.pop()
                buffer_data.append(result[0])
                buffer_gene_id.append(result[1])

            elif self.arity_list[i] == 'output': #Output the result of computation.
                if self.nan_inf_check(buffer_data[-1]):
                    return None, None
                elif len(buffer_data) == 1:
                    return buffer_data[0], buffer_gene_id[0]
                else:
                    print('Computation Error!!!')