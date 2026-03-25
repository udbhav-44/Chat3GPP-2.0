import os
import subprocess

def convert_to_html(content, output_base="pathway", output_dir=None):
    write_artifacts = os.getenv("WRITE_UI_ARTIFACTS", os.getenv("WRITE_ARTIFACTS", "false")).lower() in {
        "1",
        "true",
        "yes",
    }
    if not write_artifacts:
        return ""
    output_root = output_base
    if output_dir:
        output_root = os.path.join(output_dir, output_base)

    md_path = f"{output_root}_report.md"
    html_path = f"{output_root}.html"

    # Save content to a markdown file
    with open(md_path, 'w') as file:
        file.write(content)

    try:
        # Execute the command
        result = subprocess.run(
            ['grip', md_path, '--export', html_path],  # Command and arguments as a list
            check=True,                                   # Raise an exception if the command fails
            capture_output=True,                          # Capture the command's output
            text=True                                     # Ensure output is in text format, not bytes
        )
        print("Command executed successfully!")
        print(result.stdout)  # Print any output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    except FileNotFoundError:
        print("The 'grip' command was not found. Ensure it is installed and available in your PATH.")

    return html_path
