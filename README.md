# ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization
 
ViSoLex is an open-source project for normalizing Non-Standard Words (NSWs) in Vietnamese social media texts. This repository offers a weakly supervised framework for lexical normalization and includes methods for training, inference, and NSW detection. The system supports customizable models, including ViSoBERT and BARTpho.

## Repository Structure

```
data/                                    # Contains data files for training and evaluation
dict/                        
  └── dictionary.json                    # NSW dictionary with detailed GPT-4o interpretations
experiments/                             # Checkpoints available via Google Drive
framework_components/
  ├── aligned_tokenizer.py               # Token-level alignment tokenization
  ├── data_handler.py                    # Data loading and management
  ├── evaluator.py                       # Evaluation metrics
  ├── log.py                             # Logging system
  ├── rule_attention_network.py          # Rule attention network for training and inference
  ├── student.py                         # Methods for student model, links to lexical normalization
  └── teacher.py                         # Methods for teacher model, links to rule attention network
interface/
  ├── static/
  │   ├── index.css                      # CSS for the UI
  │   └── script.js                      # JavaScript for the UI
  └── index.html                         # Front-end for the web application
normalizer/                              # Lexical normalization models
  ├── model_construction/
  │   ├── bartpho.py                     # BARTpho for lexical normalization
  │   ├── nsw_detector.py                # Binary predictor for NSW detection
  │   ├── phobert.py                     # PhoBERT for lexical normalization (not used in research)
  │   └── visobert.py                    # ViSoBERT for lexical normalization
  ├── trainer.py                         # Main training class
  ├── trainer_methods.py                 # Reusable training methods
  └── trainer_tools.py                   # Auxiliary training tools and utilities
app.py                                   # Flask application for UI
arguments.py                             # Define command arguments
chatgpt.py                               # Run OpenAI API for NSW lookup
demo.ipynb                               # Colab notebook for simple demo
demo.py                                  # Terminal demo
main.py                                  # Run experiments, including training and evaluation
project_variables.py                     # Define global variables
requirements.txt                         # System dependencies
utils.py                                 # Supporting functions for preprocessing, result writing, etc.
```
#### **Note:**
- Experimental data is available at this GitHub repository.
- Model checkpoints can be downloaded via this [Google Drive URL](https://drive.google.com/drive/folders/1soK6OtsJ5L2C0N1nJMaVDEfySZ7FDfil?usp=drive_link).

## Instructions for Researchers and Developers

### 1. Hardware Requirements

- **ViSoBERT**: Vocabulary size 15,004  
  Minimum: 55GB CPU RAM, 12GB GPU RAM
- **BARTpho**: Vocabulary size 43,000  
  Minimum: 120GB CPU RAM, 32GB GPU RAM

### 2. Installation and Training

To retrain the models, reproduce results, and evaluate performance, follow these steps:

#### Step 1: Create and Activate Conda Environment
```bash
conda create -n visolex python=3.10
conda init
bash
conda activate visolex
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Run Training Command
```bash
python main.py --student_name visobert --training_mode weakly_supervised --num_epochs 5 --num_unsup_epochs 5 --eval_batch_size 16 --unsup_batch_size 16 --num_iter 10 --lower_case --hard_student_rule --soft_labels --append_n_mask --nsw_detect --rm_accent_ratio 1.0
```

For detailed explanations of command arguments, refer to `arguments.py`.

### 3. Quick Demo

After setting up the environment and installing dependencies, you can run a quick demo on the terminal:

```bash
python demo.py
```

Alternatively, use the provided Colab notebook: `demo.ipynb`.

## Instructions for Non-Experts

ViSoLex provides a user-friendly interface for non-technical users. You can run the Flask web application as follows:

#### Step 1: Set Up the Environment and Install Dependencies

Same as above.

#### Step 2: Run the Flask App

```bash
!python app.py
```

This web interface can also be deployed on Google Colab. See the tutorial in `demo.ipynb`.

## Video Tutorial

A demonstration video on how to use the interface is accessible via this [Google Drive URL](https://drive.google.com/file/d/1JJ3JqrXBan_0KjLFHDjxYwZ1MRkU61ou/view?usp=drive_link).

## License

This project is licensed under the MIT License.
