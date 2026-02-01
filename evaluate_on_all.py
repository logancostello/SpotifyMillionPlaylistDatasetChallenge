import subprocess
import sys

for i in range(1, 11):
    subprocess.run(
        [sys.executable, "create_train_test_split.py", f"{i}"],
        check=True
    )

    subprocess.run(
        [sys.executable, "evaluate_model.py"],
        check=True
    )
