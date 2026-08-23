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
