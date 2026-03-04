# Templates Folder — CertifyX

Place **genuine certificate images** here so the system can compare
uploaded certificates against them using SSIM structural similarity.

## How it works
When a certificate is uploaded, CertifyX computes SSIM similarity
against every image in this folder. If the best match score is below
0.60, the verdict is overridden to **SUSPICIOUS**.

## What to put here
- Scanned or photographed genuine certificates you trust
- One image per unique certificate template/format
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff

## Naming convention
Use descriptive names, e.g.:
  - vtu_degree_template.jpg
  - cbse_marksheet_template.jpg
  - anna_university_template.png

## Notes
- If this folder is EMPTY, the template check is SKIPPED (no penalty)
- The system resizes all templates to match the uploaded image before comparing
- Aim for at least one representative template per institution you want to validate
