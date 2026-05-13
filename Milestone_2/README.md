# IFT6758 Project Milestone 2

This branch contains the code and assets for the IFT6758 NHL Play-by-Play Analysis project Milestone 2.  
It includes both the main analytical notebook (`milestone2.ipynb`) and a Jekyll-based blog (`myblog`) used to showcase results.

The project demonstrates key data science concepts such as:
- Extracting and processing NHL play-by-play data
- Using Feature Engineering and Machine Learning to analyze key metrics used to predict expected goals
- Continue with visualizations and other statistical analysis of influential features on scoring hockey goals 
- Building reproducible environments for collaboration and deployment
- Using WANDB to track all experiments and images outputed in the key. The WANDB API key to enter when prompted in the code to enable this tracking is **50c6486ca894b323e5061c4513b477036ac87f8a**

**milestone2 overview:**
Each folder within the milestone2 branch serves a specific purpose:

- **`src/`**: Main branch folder containing code, blog content, and other keys files  
  - **`myblog/`**: Contains `_posts/`, configuration files, and assets for the Jekyll site  
  - **`milestone2.ipynb/`**: Contains main code file for Milestone 2   
  - **`requirements.txt`**: Text file which contains all requirements for packages and libraries needed to run milestone2.ipynb

- **`README.md`** — This file
- **`environments.yml`** - Defines all the dependencies and Python version needed to recreate your project’s Conda environment.
- **`gitignore.txt`** - Tells Git which files and folders to skip when tracking or committing changes.
- **`setup.py`** - Specifies how to install your project as a Python package, including its dependencies and metadata.


## Set up Instructions (Python version == 3.13.3)

Follow these steps to reproduce the environment and run both the notebook and the Jekyll blog locally. 

    git clone https://github.com/LuciusH1998/IFT-6758-Group-8-Milestone-1.git
    cd IFT-6758-Group-8-Milestone-1
    cd src

## Create and Activate a virtual Environment

We’ll use venv for simplicity. From inside the src directory:

    python -m venv venv
    source venv/bin/activate     # On macOS/Linux
    venv\Scripts\activate        # On Windows

## Install Dependencies

    pip install -r requirements.txt
    
This will install all Python dependencies required for the notebook and analysis.

### Run the milestone2.ipynb notebook

Launch Jupyter Lab or Notebook and open milestone1.ipynb:

    jupyter lab
    
   or
   
    code .

After making sure that the newly created virtual environment has been selected as the kernel, execute all cells to reproduce the results and visualizations.

### Serve the Jekyll Blog Locally

To preview the project blog locally, navigate to the blog folder and start the Jekyll server:
    
    cd src
    cd myblog
    bundle exec jekyll serve

Once it starts, open your browser and go to:

http://127.0.0.1:4000

or directly to your post:

http://127.0.0.1:4000/data%20science/nhl/python/2025/11/07/nhl-play-by-play-analysis-milestone-2.html
