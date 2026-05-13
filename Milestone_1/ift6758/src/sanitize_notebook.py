import nbformat

path = "milestone1.ipynb"
nb = nbformat.read(path, as_version=4)

# Remove problematic global widget metadata if it exists
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]
    print("🧹 Removed broken 'widgets' metadata from notebook-level metadata.")

# Clean cell-level widget metadata
for cell in nb.cells:
    if "metadata" in cell:
        cell.metadata.pop("widgets", None)
        cell.metadata.pop("widget_state", None)
        cell.metadata.pop("application/vnd.jupyter.widget-view+json", None)

# Save cleaned notebook
nbformat.write(nb, path)
print(" Notebook cleaned and saved! Try reuploading to GitHub now.")
