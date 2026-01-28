from pathlib import Path

ROOT = Path('C:/Users/heson/Desktop/PROBLEM_B 2026.1.26/sustainable_tourism_juneau')
md_files = list(ROOT.glob('**/*.md'))

# placeholder for $$ masking
DD = '<<DOLLAR_DOLLAR_PLACEHOLDER_42>>'

for md in md_files:
    text = md.read_text(encoding='utf-8')
    new_text = text

    # Convert display delimiters \[ ... \] -> $$ ... $$
    if '\\[' in new_text or '\\]' in new_text:
        new_text = new_text.replace('\\[', '$$').replace('\\]', '$$')

    # Convert inline delimiters \( ... \) -> $ ... $
    if '\\(' in new_text or '\\)' in new_text:
        new_text = new_text.replace('\\(', '$').replace('\\)', '$')

    # Mask $$ to protect them during single-$ cleanup
    if '$$' in new_text:
        new_text = new_text.replace('$$', DD)

    # Trim spaces directly after an opening $ and before a closing $
    # Also collapse accidental newlines inside inline $...$ into no-newline
    new_text = new_text.replace('$ ', '$').replace(' $', '$').replace('$\n', '$').replace('\n$', '$')

    # Restore $$ placeholders
    if DD in new_text:
        new_text = new_text.replace(DD, '$$')

    if new_text != text:
        md.write_text(new_text, encoding='utf-8')
        print(f'updated: {md}')
    else:
        print(f'skip: {md}')


