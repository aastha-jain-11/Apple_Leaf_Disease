# Apple Disease Severity Annotation Rulebook

## Purpose

Create pixel-level semantic segmentation masks for severity estimation.

Each image is annotated for:

- `leaf`
- `lesion`

`background` is implicit and will be generated automatically during mask conversion.

### Mask class IDs

| Class | Pixel ID |
|---|---:|
| background | 0 |
| leaf | 1 |
| lesion | 2 |

## What to annotate

### 1. Leaf

Draw a polygon around the visible leaf tissue.

Include:
- normal leaf tissue
- visibly diseased leaf tissue
- leaf veins when they are part of the leaf

Exclude:
- gray/purple background
- table/background surface
- objects that are not leaf tissue

### 2. Lesion

Draw polygons around tissue that is visibly affected by the disease.

The objective is to measure **diseased area**, not to count every individual spot perfectly.

Include:
- clearly diseased/discolored tissue
- connected or merged lesion regions
- visibly affected areas inside the leaf boundary

Do not include:
- normal leaf tissue
- shadows caused only by lighting
- camera/specular highlights
- ordinary veins with no visible disease
- background
- obvious mechanical damage when it cannot reasonably be interpreted as disease

### 3. Background

Do NOT draw a background polygon.

Any pixel outside the leaf polygon becomes background (class 0) automatically.

## Disease-specific guidance

### Apple scab

Annotate clearly visible scab lesions and diffuse diseased regions.

If several scab lesions have merged into one continuous diseased region, annotate the entire visibly affected region as one lesion region.

Do not artificially split merged lesions.

### Black rot

Annotate the visibly diseased tissue, including clearly defined circular/irregular lesion regions.

If a lesion has a darker center and a lighter diseased boundary, include the full visibly affected diseased region rather than only the darkest center.

### Cedar apple rust

Rust lesions can be small and subtle.

Prioritize **visible diseased area** over trying to outline tiny individual spots perfectly.

Do not label yellow/orange coloration as lesion unless it is visually consistent with disease rather than lighting or normal leaf variation.

### Healthy

Draw the leaf polygon.

Do not draw any lesion polygons.

The resulting mask should therefore contain:
- leaf = 1
- background = 0
- lesion = 0

This gives severity index = 0%.

## Polygon rules

1. Use polygons rather than rough rectangles.
2. Follow the visible leaf boundary reasonably closely.
3. Do not spend excessive time tracing single-pixel edges.
4. For merged lesions, use one continuous polygon when practical.
5. If two lesions are clearly separate, separate polygons are acceptable.
6. Lesion polygons may overlap the leaf polygon. The conversion script gives lesion pixels priority, so overlap is safe.
7. Stay consistent across images.

## Uncertain cases

Do not invent disease regions.

If an area is genuinely ambiguous:
- inspect the image at higher zoom
- compare with nearby tissue
- if still uncertain, mark the image for review rather than making an arbitrary annotation

## Quality-control checklist

Before saving each annotation:

- [ ] Entire visible leaf is covered by `leaf`
- [ ] Background is not labeled as leaf
- [ ] All clearly visible diseased tissue is labeled `lesion`
- [ ] Shadows are not mistaken for lesions
- [ ] Normal veins are not mistaken for lesions
- [ ] Lesion polygons do not extend outside the leaf
- [ ] Healthy images have no lesion polygons
- [ ] Annotation is saved successfully as LabelMe JSON

## Severity calculation

After mask conversion:

`Severity Index (%) = (lesion_pixels / leaf_pixels) × 100`

Segmentation metrics such as Dice and IoU evaluate the segmentation model. They are not part of the severity-index formula.

## Important

The `low`, `moderate`, and `high` values in `severity_selection.csv` are **candidate-selection labels only**. They are not the final severity ground truth.

Final continuous severity will come from the annotated leaf/lesion masks.

Six-grade thresholds should be finalized only after examining the measured severity-index distribution and validating the grading scale.
