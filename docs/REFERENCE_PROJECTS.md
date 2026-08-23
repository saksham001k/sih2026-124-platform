# Reference-project policy

The first commit accidentally stored eleven cloned repositories as Git gitlinks without a
`.gitmodules` file. A fresh clone therefore contained empty, non-initializable folders.
Those broken entries were removed.

Before adapting an external project:

1. Record its canonical URL and exact commit.
2. Verify its software and model-weight licences.
3. Add only the component needed by this project.
4. Preserve attribution and licence notices.
5. Add an integration test proving that the adapted component works.

This especially applies to ANPR, pothole-detection and traffic-analysis references.

## Licensed dependencies integrated in DrishtiPath

| Component | Version | Licence | Repository |
|---|---|---|---|
| FastALPR | 0.4.0 | MIT | https://github.com/ankandrew/fast-alpr |
| open-image-models | 0.6.0 | MIT | https://github.com/ankandrew/open-image-models |
| fast-plate-ocr | 1.1.0 | MIT | https://github.com/ankandrew/fast-plate-ocr |

DrishtiPath consumes these packages as runtime dependencies. It does **not** copy code or
weights from the Knight-Sight reference repository.

### Default ONNX models

| Role | Model identifier |
|---|---|
| Plate detector | `yolo-v9-t-384-license-plate-end2end` |
| Plate OCR | `cct-xs-v2-global-model` |
| Execution provider | `CPUExecutionProvider` |

Model files are downloaded by the FastALPR runtime on first use and must remain outside Git.
