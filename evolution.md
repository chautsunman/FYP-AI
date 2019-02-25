# Steps
1. Get the latest evolution.ipynb (`git pull origin evolution`).
2. Upload evolution.ipynb to your working Google Drive folder, rename for different input options if needed.
3. Open evolution.ipynb with Google Colaboratory.
4. Run the setup Google Colaboratory cell to set up the folder structure in Google Colaboratory’s VM.
5. Upload model.py and dnn_regression.py in models to the models folder.
6. Upload all required data CSV to data/stock_prices folder.
7. Upload build_dataset.py and options.py.
8. Run all import dependencies cells.
9. Change `input_options` in initialize models section to working input options (refer to input options document).
10. Change `stock_code` in initialize models section to working stock code.
11. Change `ITERATIONS` in Evolution algorithm section based on available time.
12. Run all cells in “Get last run data”, “Initialize models”, “Initialize errors”, “Evolution algorithm” and “Write this run data” sections.
13. last_run.json, evolution_tensorboard_logs.zip and evolution_model_graphs.zip will be generated after running cells in “Write this run data” section.
14. If time allows, re-run all cells, starting from “Get last run data” section, for another run of `ITERATIONS` evolution iterations. New last_run.json and 2 zips will be generated, with data from all previous runs.
15. Download last_run.json, evolution_tensorboard_logs.zip and evolution_model_graphs.zip for next time.
16. Repeat all steps, with an extra step of uploading last_run.json next time.
17. last_run.json will be appended with new data, download and replace the old file. evolution_tensorboard_logs.zip and evolution_model_graphs.zip will have data only from a particular Google Colaboratory session, download them with another name and manually merge the content.
18. Upload last_run.json, merged evolution_tensorboard_logs.zip and merged evolution_model_graphs.zip to Google Drive.
19. Run all cells in “Plot predictions” and “Plot evolution data” section.
20. Repeat all steps for another input options.
