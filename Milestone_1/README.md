# IFT6758 Project Milestone 1

This repository contains the code and assets for the IFT6758 NHL Play-by-Play Analysis project.  
It includes both the main analytical notebook (`milestone1.ipynb`) and a Jekyll-based blog (`myblog`) used to showcase results.

The project demonstrates key data science concepts such as:
- Extracting and processing NHL play-by-play data  
- Visualizing hockey rink and game statistics  
- Building reproducible environments for collaboration and deployment  
<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

**Directory overview:**
Each folder serves a specific purpose:

- **`ift6758/`**: Main package directory containing source files, notebooks, and blog content  
  - **`src/`**: Core working directory for scripts, data, and Jupyter notebooks  
  - **`myblog/`**: Contains `_posts/`, configuration files, and assets for the Jekyll site  
  - **`_site/`**: Automatically generated output folder created by the Jekyll build process  

- **`README.md`** — This file
- **`environments.yml`**
- **`gitignore.txt`**


## Set up Instructions (Python version == 3.13.3)

Follow these steps to reproduce the environment and run both the notebook and the Jekyll blog locally. 

    git clone https://github.com/LuciusH1998/IFT-6758-Group-8-Milestone-1.git
    cd IFT-6758-Group-8-Milestone-1
    cd ift6758
    cd src


## Create and Activate a virtual Environment

We’ll use venv for simplicity. From inside the src directory:

    python -m venv venv
    source venv/bin/activate     # On macOS/Linux
    venv\Scripts\activate        # On Windows

## Install Dependencies

    pip install -r requirements.txt
    
This will install all Python dependencies required for the notebook and analysis.

### Run the milestone1.ipynb notebook

Launch Jupyter Lab or Notebook and open milestone1.ipynb:

    jupyter lab
    
   or
   
    code .

After making sure that the newly created virtual environment has been selected as the kernel, execute all cells to reproduce the results and visualizations.

### Serve the Jekyll Blog Locally

To preview the project blog locally, navigate to the blog folder and start the Jekyll server:

    cd myblog
    bundle exec jekyll serve

Once it starts, open your browser and go to:

http://127.0.0.1:4000

or directly to your post:

http://127.0.0.1:4000/data%20science/nhl/python/2025/10/04/nhl-play-by-play-analysis.html

