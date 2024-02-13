from trainers.base_trainer import SimpleTrainer
import numpy as np
import os, re
import wandb
from supar import Parser
import sys 
import string
import heapq
sys.path.append("..") 
import random
import utils.nat_inst_gpt3 as gpt3
import utils.nat_inst_gpt2 as gpt2
import utils.nat_inst_phi2 as phi2
import utils.nat_inst_tinyllama as tinyllama
from pathlib import Path

class GAC_trainer(SimpleTrainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, num_tournaments, backbone):
        super(GAC_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone)
        self.num_tournaments = num_tournaments
        self.patience_counter = 1
        self.W_candidates = []
        self.W_scores = []
        self.original_candidate = None
        self.original_score = None
        self.result_candidate = None
        self.result_score = None
        self.parser = Parser.load('crf-con-en')
        self.para_tokenizer = None
        self.para_model = None
        self.W_deletesets = []
        self.state = {}
        
    def get_state(self, current_iteration, delete_tracker):
        self.state = {'np_random_state' : np.random.get_state(), 'random_state' : random.getstate(), 'current_iteration' : current_iteration, 'W_candidates' : self.W_candidates, 'W_scores' : self.W_scores, 'result_candidate' : self.result_candidate, 'result_score' : self.result_score, 'patience_counter': self.patience_counter, 'delete_tracker' : delete_tracker}
        
    def set_state(self):
        current_iteration = self.state['current_iteration'] 
        delete_tracker = self.state['delete_tracker']
        self.W_candidates = self.state['W_candidates'] 
        self.W_scores = self.state['W_scores'] 
        self.result_candidate = self.state['result_candidate']
        self.result_score =  self.state['result_score']
        self.patience_counter = self.state['patience_counter']
        np.random.set_state(self.state['np_random_state'])
        random.setstate(self.state['random_state'])
        return current_iteration, delete_tracker

    def tournament_selection(self):
        S_candidates = []
        S_scroes = []
        for k in range(self.num_tournaments):  
            parent = np.random.randint(0,len(self.W_candidates))  # parent, score_parent <-- Random(W)
            S_candidates.append(self.W_candidates[parent])  # S_candidate = S_candidate + parent 
            S_scroes.append(self.W_scores[parent])  # S_score = S_score + score_parent
        base_idx = np.argmax(S_scroes)   # base_idx = \arg max_{idx \in S} S_score
        base_candidate = S_candidates[base_idx] # base <-- S_candidates(base_idx)
        base_score = S_scroes[base_idx] # base_score <-- S_candidates(base_idx)
        
        return base_candidate, base_score
    
    def crossover(self, parent_1, parent_2, args):
        flag_error = False
        try:
            phrases_1_pun = self.get_phrase_lookup_pun(parent_1, args)
            phrases_2_pun = self.get_phrase_lookup_pun(parent_2, args)
        except AttributeError:
            offspring=''
            flag_error = True
            return offspring, flag_error
        phrases_1 = [phrase for phrase in list(phrases_1_pun.values()) if phrase not in string.punctuation or phrase == '']
        phrases_2 = [phrase for phrase in list(phrases_2_pun.values()) if phrase not in string.punctuation or phrase == '']

        split = np.random.randint(0,max(len(phrases_1),len(phrases_2)))
        
        if split >= len(phrases_1):
            offspring_phrases = list(phrases_1_pun.values()) + list(phrases_2_pun.values())[list(phrases_2_pun.values()).index(phrases_2[split]):]
        elif split >= len(phrases_2):
            offspring_phrases = list(phrases_1_pun.values())[0:list(phrases_1_pun.values()).index(phrases_1[split])]
        else:
            offspring_phrases = list(phrases_1_pun.values())[0:list(phrases_1_pun.values()).index(phrases_1[split])] + list(phrases_2_pun.values())[list(phrases_2_pun.values()).index(phrases_2[split]):]
        offspring_words = []
        for phrase in offspring_phrases:
            offspring_words = offspring_words + self.word_tokenize(phrase)
        offspring = self.detokenize(offspring_words)
        return offspring, flag_error

    def containenglish(self, str0):
        return bool(re.search('[a-z A-Z]', str0))

    def mutated(self, base_candidate, phrase_lookup, use_add, i, edit_operations, args):

        
        if base_candidate == self.original_candidate:
            for p in phrase_lookup.values(): print(p)
        if use_add: 
            if len(self.W_deletesets[i]): 
                if 'add' not in edit_operations: edit_operations.append('add')
            else: 
                if 'add' in edit_operations: edit_operations.remove('add')

        empty = True
        while empty:
            if self.num_compose == 1:
                edits = np.random.choice(edit_operations, self.num_candidates)
            else: 
                edits = []
                for n in range(self.num_candidates):
                    edits.append(np.random.choice(edit_operations, self.num_compose))
            print(edits)

        # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str): 
                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, self.W_deletesets[i])
                    empty = not self.containenglish(candidate)
                    if not empty:
                        print(candidate)
                        candidates.append(candidate)
                        deleteset = []
                        if edit  == 'del': 
                            deleteset = self.W_deletesets[i] + [phrase_lookup[indices[0]]]
                        if edit == 'add': 
                            if len(indices): 
                                deleteset = self.W_deletesets[i]
                                deleteset.remove(indices[0])
                        self.W_deletesets.append(deleteset)
                    else:
                        print('''Note: The mutated candidate is an empty string, and it is deleted.''')
                else:
                    old_candidate = base_candidate
                    composed_deletes = []
                    composed_adds = []
                    for op in edit:
                        phrase_lookup = self.get_phrase_lookup(old_candidate, args)
                        new_candidate, indices = self.perform_edit(op, old_candidate, phrase_lookup, self.W_deletesets[i])
                        empty = not self.containenglish(new_candidate)
                        if not empty:
                            print(new_candidate)
                            if op  == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                            if op == 'add': 
                                if len(indices): composed_adds.append(indices[0])
                            old_candidate = new_candidate
                        else:
                            break

                    if not empty:
                        candidates.append(new_candidate)
                        deleteset = []
                        if 'del' in edit: 
                            deleteset = self.W_deletesets[i] + composed_deletes
                        if 'add' in edit and len(composed_adds) > 0: 
                            deleteset = self.W_deletesets[i]
                            for phrase in composed_adds:
                                deleteset.remove(phrase) 
                        self.W_deletesets.append(deleteset)
        scores = []
        for c, candidate in enumerate(candidates):
            scores.append(self.score(candidate, args=args))
            print(scores[-1])

        return candidates, scores  

    def train(self, instruction, chosen_task_name, args):
        meta_path = os.path.join(args.meta_dir, args.meta_name)
        meta_file = open(meta_path, 'w+')

        edit_operations = args.edits

        use_add = 'add' in edit_operations

        if 'sub' in edit_operations:
            self.if_sub(edit_operations)

        self.init_population(instruction, args)

        self.W_candidates = self.W_candidates * args.population_size 
        self.W_scores = self.W_scores * args.population_size  
        for i in range(args.population_size):
            self.W_deletesets.append([])   # record a DeleteSet(D) for each prompt in the population     

        meta_file.write("Original Candidate:\t "+ self.original_candidate + '\n')
        meta_file.write("Original Score:\t "+ str(self.original_score) + '\n')
        meta_file.write("\n")
        wandb.log({"original_score": self.original_score})
        current_iteration = 0 
        delete_tracker = []

        if len(args.resume):
            print("Resuming the searching from checkpoints...")
            self.load(args.resume)
            current_iteration, delete_tracker = self.set_state()
            
        while current_iteration < self.maxiter:
            current_iteration += 1
            for j in range(args.num_offspring):

                parent_1, parent_score_1 = self.tournament_selection() # parent_1, parent_score_1 <-- tournament(W_{i-1})
                parent_2, parent_score_2 = self.tournament_selection() # parent_1, parent_score_1 <-- tournament(W_{i-1})
                meta_file.write("parent_1" + str(j) + ":\t " + parent_1+ '\n')
                meta_file.write("parent_2" + str(j) + ":\t " + parent_2+ '\n')

                #################### zhushi
                empty = True

                while empty:

                    offspring, flag_error = self.crossover(parent_1, parent_2, args) # offspring <-- crossover(parent_1, parent_2)

                    empty = not self.containenglish(offspring) #### 

                    if flag_error:
                        break
                ####################

                if flag_error: 
                    meta_file.write("AttributeError occurs (parser) and skip this crossover operation"+ '\n')
                    print('AttributeError occurs (parser) and skip this crossover operation')
                    continue

                meta_file.write("Offspring" + str(j) + ":\t " + offspring + '\n')
                if offspring not in self.W_candidates:
                    offspring_score = self.score(offspring, args=args)
                    # population W  <-- offspring
                    self.W_candidates.append(offspring) 
                    self.W_scores.append(offspring_score) 
                    self.W_deletesets.append(self.W_deletesets[self.W_candidates.index(parent_1)])
                    meta_file.write("Offspring:\t" + offspring + '\n') 
                    meta_file.write("Adding offspring in the population" + '\n') 

            if args.num_offspring > 0:
                top_N_p_idx_list = heapq.nlargest(args.population_size, range(len(self.W_scores)), self.W_scores.__getitem__)
                # population W <-- Top-N_p(population W)
                W_candidates_top_N_p = [self.W_candidates[i] for i in top_N_p_idx_list]
                W_scores_top_N_p = [self.W_scores[i] for i in top_N_p_idx_list]
                W_deletesets_N_p = [self.W_deletesets[i] for i in top_N_p_idx_list]
                self.W_candidates = W_candidates_top_N_p
                self.W_scores = W_scores_top_N_p
                self.W_deletesets = W_deletesets_N_p
                self.update_result(self.W_candidates, self.W_scores) # update result

                wandb.log({"result_score_after_crossover": self.result_score})

            if self.patience_counter > args.patience:
                print('Ran out of patience')
                meta_file.write('Ran out of patience \n')
                break
            else : pass

            # W' <-- W
            W_candidates_m = self.W_candidates
            W_scores_m = self.W_scores
            for i, base_candidate in enumerate(W_candidates_m):
                if args.mutation_prob > np.random.random():            
                    try:
                        phrase_lookup = self.get_phrase_lookup(base_candidate, args)
                    except AttributeError:
                        self.W_scores.remove(self.W_scores[self.W_candidates.index(base_candidate)]) 
                        self.W_candidates.remove(base_candidate)
                        meta_file.write("AttributeError occurs (parser) and skip this mutation"+ '\n')
                        print('AttributeError occurs (parser) and skip this mutation')
                        continue
                    candidates, scores = self.mutated(base_candidate, phrase_lookup, use_add, i, edit_operations, args)

                    W_scores_m = W_scores_m + scores
                    W_candidates_m = W_candidates_m + candidates
                    W_deletesets_m = self.W_deletesets

            top_N_p_idx_list_m = heapq.nlargest(args.population_size, range(len(W_scores_m)), W_scores_m.__getitem__)
            W_candidates_m_top_N_p = [W_candidates_m[i] for i in top_N_p_idx_list_m]
            W_scores_m_top_N_p = [W_scores_m[i] for i in top_N_p_idx_list_m]
            W_deletesets_m_top_N_p = [W_deletesets_m[i] for i in top_N_p_idx_list_m]

            self.W_candidates = W_candidates_m_top_N_p
            self.W_scores = W_scores_m_top_N_p
            self.W_deletesets = W_deletesets_m_top_N_p

            self.update_result(self.W_candidates, self.W_scores)

            wandb.log({"result_score_after_mutation": self.result_score})

            self.result_candidate = self.detokenize(self.word_tokenize(self.result_candidate))

            if current_iteration % args.checkpoint_freq == 0:
                self.get_state(current_iteration, delete_tracker)
                ckpt_dir = Path(args.output_dir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                filename = "task{}_step{}.pickle".format(args.task_idx, current_iteration-1)
                ckpt_path = ckpt_dir / filename
                self.save(ckpt_path)
                
            if args.backbone == "gpt3":
                count = gpt3.complete_gpt3.count
        
            if args.backbone == "gpt2":
                count = gpt2.complete_gpt2.count

            if args.backbone == "phi2":
                count = phi2.complete_gpt2.count

            if args.backbone == "tinyllama":
                count = tinyllama.complete_gpt2.count
            
            # if count >= args.budget:
            #     print('Ran out of budget')
            #     break
            # else: 
            #     continue

            if self.patience_counter > args.patience:
                print('Ran out of patience')
                meta_file.write('Ran out of patience \n')
                break
            elif count >= args.budget:
                print('Ran out of budget')
                break
            else: 
                continue

        wandb.log({"result_score": self.result_score})
        
        if args.backbone == "gpt3":
            count = gpt3.complete_gpt3.count

        if args.backbone == "llama":
            count = gpt3.complete_llama2_7b.count
        
        if args.backbone == "gpt2":
            count = gpt2.complete_gpt2.count

        if args.backbone == "phi2":
            count = phi2.complete_gpt2.count

        if args.backbone == "tinyllama":
            count = tinyllama.complete_gpt2.count
            
        print('APICalls for search:\t', count)

        wandb.log({"apicalls_search": count})
        meta_file.write('\n')
        searched_score = self.test(self.result_candidate, args)

        meta_file.write('Testing .... \n')
        if args.print_orig:
            print('Task:\t', chosen_task_name)
            print('Original Instruction:\t', self.original_candidate)
            orig_score = self.score(self.original_candidate, 'test', args=args)
            print('Original Accuracy:\t', str(orig_score))
            meta_file.write('Original Accuracy:\t'+ str(orig_score)+ '\n')

        if self.result_candidate == self.original_candidate: 
            print('No viable candidate found!')
            meta_file.write('No viable candidate found!\n')
            print('APICalls:\t', count)
            meta_file.write('APICalls:\t'+ str(count) + '\n')
            wandb.log({"Original Accuracy": orig_score})
            exit()

        wandb.log({"searched_accuracy": searched_score})
        wandb.log({"apicalls_total": count})

        print('Accuracy after search:\t', str(searched_score))
        print('Instruction after search:\t', self.result_candidate)
        meta_file.write('Instruction after search:\t'+ self.result_candidate+ '\n')
        meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
        print('APICalls:\t', count)
        meta_file.write('APICalls:\t'+ str(count) + '\n')

        wandb.save(meta_path)

    def test(self, instruction, args):

        print('\nTesting .... ')

        searched_score = self.score(instruction, 'test', write=args.write_preds, args=args)

        return searched_score