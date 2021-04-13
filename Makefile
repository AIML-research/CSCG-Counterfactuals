MY_SRC_DIR := src
COMPETITOR_DIR := competitor/synth-action-seq
RESULTS_DIR := results
PLOTS_DIR := plots
EXPERIMENT_INSTANCES_DIR := experiment_data/test_instances


all: requirements experiments analysis

# Install all libraries etc.
requirements: requirements.txt
	pip install -r requirements.txt 


# Run all experiments
experiments: clean_experiment_results experiment_setup cscf scf competitor

# Create initial instances
experiment_setup: experiment_setup_adult experiment_setup_german


# Run SCF on all
cscf: cscf_adult

# Run SCF on all
scf: scf_adult scf_german

# Run competitor on all
competitor: competitor_adult competitor_german




# Create initial instances
experiment_setup_adult:
	python $(MY_SRC_DIR)/create_experiment_instances.py adult
	cp $(EXPERIMENT_INSTANCES_DIR)/processed_adult_n=100.csv $(COMPETITOR_DIR)/

# Create initial instances
experiment_setup_german:
	python $(MY_SRC_DIR)/create_experiment_instances.py german
	cp $(EXPERIMENT_INSTANCES_DIR)/processed_german_n=100.csv $(COMPETITOR_DIR)/


# Run CSCF on adult
cscf_adult: 
	python $(MY_SRC_DIR)/evaluation.py with comp_adult_problem_alternative_cfg
	rm -r $(RESULTS_DIR)/my_method/adult_cscf/
	mkdir $(RESULTS_DIR)/my_method/adult_cscf/
	find $(dir $(RESULTS_DIR)/*adult/Final*/) -type f -exec cp {}  $(RESULTS_DIR)/my_method/adult_cscf/ \;
	rm -r $(dir $(RESULTS_DIR)/*adult/)

# Run SCF on adult
scf_adult: 
	python $(MY_SRC_DIR)/evaluation.py with comp_adult_problem_main_cfg
	rm -r $(RESULTS_DIR)/my_method/adult_scf/
	mkdir $(RESULTS_DIR)/my_method/adult_scf/
	find $(dir $(RESULTS_DIR)/*adult/Final*/) -type f -exec cp {}  $(RESULTS_DIR)/my_method/adult_scf/ \;
	rm -r $(dir $(RESULTS_DIR)/*adult/)

# Run SCF on german
scf_german: 
	python $(MY_SRC_DIR)/evaluation.py with comp_german_problem_main_cfg
	rm -r $(RESULTS_DIR)/my_method/german_scf/
	mkdir $(RESULTS_DIR)/my_method/german_scf/
	find $(dir $(RESULTS_DIR)/*german/Final*/) -type f -exec cp {}  $(RESULTS_DIR)/my_method/german_scf/ \;
	rm -r $(dir $(RESULTS_DIR)/*german/)


# Run competitor on adult
competitor_adult: experiment_setup_adult
	python $(COMPETITOR_DIR)/run.py --target-model adult --ckpt model.h5 --mode vanilla --exp-name comp_exp --l 2
	rm -r $(RESULTS_DIR)/competitor/adult/
	mkdir $(RESULTS_DIR)/competitor/adult/
	find $(dir $(COMPETITOR_DIR)/$(RESULTS_DIR)/*adult/) ! -name config*.json -type f -exec cp {}  $(RESULTS_DIR)/competitor/adult/ \;
	rm -r $(dir $(COMPETITOR_DIR)/$(RESULTS_DIR)/*adult/)

# Run competitor on german
competitor_german: experiment_setup_german
	python $(COMPETITOR_DIR)/run.py --target-model german --ckpt model.h5 --mode vanilla --exp-name comp_exp --l 2
	rm -r $(RESULTS_DIR)/competitor/german/
	mkdir $(RESULTS_DIR)/competitor/german/
	find $(dir $(COMPETITOR_DIR)/$(RESULTS_DIR)/*german/) ! -name config*.json -type f -exec cp {}  $(RESULTS_DIR)/competitor/german/ \;
	rm -r $(dir $(COMPETITOR_DIR)/$(RESULTS_DIR)/*german/)

# Create all plots for analysis
analysis: clean_plots
	python $(MY_SRC_DIR)/analysis/analysis.py


clean_plots:
	rm -r $(PLOTS_DIR)/
	mkdir $(PLOTS_DIR)

clean_experiment_instances:
	rm -r $(EXPERIMENT_INSTANCES_DIR)/
	mkdir $(EXPERIMENT_INSTANCES_DIR)

clean_experiment_results:
	rm -r $(RESULTS_DIR)/
	mkdir $(RESULTS_DIR)

# Removes all result files and experiment files
clean_all: clean_plots clean_experiment_instances clean_experiment_results