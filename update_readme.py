import pandas as pd

performance_results = pd.read_csv("data/model_performance.csv")

markdown_table = performance_results.to_markdown(index=False)

readme_file = "results.md"
insert_section = "## Results"

with open(readme_file, "r") as file:
    readme_content = file.read()

insert_index = readme_content.find(insert_section)
updated_readme_content = (
    readme_content[: insert_index + len(insert_section)]
    + "\n\n"
    + markdown_table
    + "\n\n"
    + readme_content[insert_index + len(insert_section) :]
)

with open(readme_file, 'w') as file:
    file.write(updated_readme_content)