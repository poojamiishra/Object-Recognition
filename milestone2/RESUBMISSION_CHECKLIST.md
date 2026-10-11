# Milestone 2 clarification checklist

Use this checklist before replacing the submitted report or sending samples to the TA.

## Already satisfied by the project

- YOLOv8 is customized for the 73 Object IDs and returns IDs plus bounding boxes.
- The generated dataset contains 4,000 multi-object images, above the 300-image minimum for a three-person group.
- Randomized training composites cover multiple layouts and object counts instead of only one fixed pattern.
- Every generated composite contains at least two objects; there are no zero-object/background-only samples.
- Train, validation, and test composites use source photos from their matching Milestone 1 split.

## Required verification to run

1. Open `Milestone2_YOLOv8_MultiObject.ipynb` in Colab with the existing Drive outputs available.
2. Run through the existing model evaluation and detection sections.
3. Run **Required standard-grid verification**.
4. Confirm `standard_grid_results.csv` contains results for both `standard_2x2` and `standard_3x3`.
5. Inspect the saved input images and confirm their dimensions:
   - `standard_2x2_input.jpg`: 448x448 with four touching 224x224 tiles.
   - `standard_3x3_input.jpg`: 672x672 with nine touching 224x224 tiles.
6. Inspect both annotated prediction images. Prefer samples marked as exact, meaning every object is correctly identified and localized with no extra detections.
7. If either standard layout performs materially worse than the randomized test set, fine-tune with standard grids added to the training mixture and rerun all evaluations. Do not replace the varied layouts with only fixed grids.

## Report update

- Add a short subsection describing the standard, gap-free grid construction.
- Report measured 2x2 and 3x3 results separately. Do not insert values until the notebook has generated them.
- Include `standard_2x2_prediction.jpg` and `standard_3x3_prediction.jpg`, with their prediction CSVs summarized below the figures.
- Keep the 4x3 stress test as an additional experiment, not as the standard-grid demonstration.
- State that there are no background-only images. Do not incorrectly claim that ordinary photo backgrounds or colored canvas areas were prohibited.
- State how each team member met the 100-image contribution requirement only after the team confirms the contribution record.

## TA sample submission

- Send one annotated 2x2 prediction and one annotated 3x3 prediction.
- Group 8 sends the samples to Nishanth on Teams.
- Include the Object IDs, confidence values, and bounding-box locations, either visibly in the images or in the accompanying CSV/table.
