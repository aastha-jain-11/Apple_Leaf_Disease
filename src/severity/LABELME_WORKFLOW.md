# LabelMe Annotation Workflow

## 1. Install LabelMe

From the project environment:

```bash
pip install labelme
```

## 2. Run the image-selection setup

From the project root:

```bash
python src/severity/setup_severity_annotation.py
```

This reads:

`outputs/severity_selection.csv`

and copies only rows marked:

- `low`
- `moderate`
- `high`

into:

```text
outputs/
└── severity/
    └── annotation_images/
        ├── apple_scab/
        ├── black_rot/
        ├── cedar_rust/
        └── healthy/
```

Excluded images are not copied.

A manifest is also created:

`outputs/severity/annotation_manifest.csv`

## 3. Annotate with LabelMe

Start LabelMe:

```bash
labelme
```

Open one class folder at a time.

For every image:

1. Create a polygon named `leaf`.
2. Trace the visible leaf boundary.
3. Create polygon(s) named `lesion` around visibly diseased tissue.
4. Do not create a `background` polygon.
5. Save the annotation as JSON.

Recommended JSON location:

```text
outputs/
└── severity/
    └── labels_json/
        ├── apple_scab/
        ├── black_rot/
        ├── cedar_rust/
        └── healthy/
```

Keep the JSON filename identical to the image filename, e.g.:

```text
image (164).JPG
image (164).json
```

## 4. Pilot first

Before annotating the full set, annotate approximately 20–30 images across all four classes.

Then review the resulting masks for consistency.

Do not train SegFormer until the pilot passes QA.

## 5. Convert JSON to masks

From the project root:

```bash
python src/severity/labelme_to_masks.py
```

The generated masks use:

- 0 = background
- 1 = leaf
- 2 = lesion

## 6. Calculate severity index

Run:

```bash
python src/severity/calculate_severity_index.py
```

This produces:

`outputs/severity/severity_ground_truth.csv`

with leaf area, lesion area, and continuous severity index.

## 7. Do not finalize grade thresholds yet

The six severity grades should be determined after the first annotated batch gives us the actual SI distribution.

The provisional scale can be reviewed later:

- Grade 0: 0%
- Grade 1: >0–10%
- Grade 2: >10–20%
- Grade 3: >20–40%
- Grade 4: >40–60%
- Grade 5: >60–100%

These are provisional until validated.
