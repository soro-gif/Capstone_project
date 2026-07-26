"""Lance un job de tuning supervisé Gemini via Vertex AI.

Prérequis :
  - Projet GCP avec facturation activée et API Vertex AI activée
  - `gcloud auth application-default login` effectué
  - train_vertex.jsonl / val_vertex.jsonl uploadés sur un bucket GCS
    (voir scripts/prepare_vertex_data.py)

Usage:
    python scripts/run_vertex_finetune.py \
        --project project-12661c0f-93ef-4036-bb0 \
        --location us-central1 \
        --train gs://billsum-finetune-project-12661c0f/train_vertex.jsonl \
        --val gs://billsum-finetune-project-12661c0f/val_vertex.jsonl \
        --name billsum-fr \
        --watch
"""
import argparse
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--location", default="us-central1")
    p.add_argument("--train", required=True, help="chemin gs:// vers train_vertex.jsonl")
    p.add_argument("--val", default=None, help="chemin gs:// vers val_vertex.jsonl")
    p.add_argument("--base-model", default="gemini-2.5-flash")
    p.add_argument("--name", default="billsum-fr")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--watch", action="store_true", help="suit le job jusqu'à la fin")
    args = p.parse_args()

    import vertexai
    from vertexai.tuning import sft

    vertexai.init(project=args.project, location=args.location)

    tuning_job = sft.train(
        source_model=args.base_model,
        train_dataset=args.train,
        validation_dataset=args.val,
        epochs=args.epochs,
        tuned_model_display_name=args.name,
    )
    print(f"Job de tuning créé : {tuning_job.resource_name}")

    if args.watch:
        while not tuning_job.has_ended:
            time.sleep(30)
            tuning_job.refresh()
            print(f"state={tuning_job.state}")
        if tuning_job.has_succeeded:
            print(f"Modèle tuné : {tuning_job.tuned_model_name}")
            print(f"Endpoint : {tuning_job.tuned_model_endpoint_name}")
        else:
            print("Échec ou annulation :", tuning_job.error)
    else:
        print(f"Suivi manuel : sft.SupervisedTuningJob('{tuning_job.resource_name}')")


if __name__ == "__main__":
    main()
