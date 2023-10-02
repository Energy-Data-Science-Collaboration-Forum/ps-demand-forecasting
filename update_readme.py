import datetime as dt
import pandas as pd

today_one = dt.datetime.now().strftime("%d %B %Y")
performance_results = pd.read_csv("data/model_performance.csv")

markdown_table = performance_results.to_markdown(index=False)

readme_file = "results.md"
insert_section = "## Results"
end_insert_section = "## Contributing"

with open(readme_file, "r") as file:
    readme_content = file.read()

insert_index = readme_content.find(insert_section)
end_insert_index = readme_content.find(end_insert_section)

updated_readme_content = (
    readme_content[: insert_index + len(insert_section)]
    + "\n\n"
    + f"The last run with the full datasets was with data up to and including {today_one} with the following results:"
    + "\n\n"
    + markdown_table
    + "\n\n"
    + readme_content[end_insert_section:]
)

with open(readme_file, 'w') as file:
    file.write(updated_readme_content)