import pandas as pd
from util.template_concepts import imagenet_templates, painting_templates, CIFAR100_classes


ip_concepts = ["snoopy", "Mickey", "SpongeBob", "Pikachu", "Hello Kitty"]
style_concepts = ["Van Gogh", "Picasso", "Monet"]


if __name__ == "__main__":
    rows = []
    idx = 1

    for concept in ip_concepts:
        for template in imagenet_templates:
            rows.append(
                {
                    "id": idx,
                    "concept": concept,
                    "prompt": template.format(concept),
                }
            )
            idx += 1

    df_ip = pd.DataFrame(rows, columns=["id", "concept", "prompt"])
    output_path = "ip_imagenet_prompts.csv"
    df_ip.to_csv(output_path, index=False)

    print(f"Saved {len(df_ip)} rows to {output_path}")

    style_rows = []
    style_idx = 1

    for concept in style_concepts:
        for template in painting_templates:
            style_rows.append(
                {
                    "id": style_idx,
                    "concept": concept,
                    "prompt": template.format(concept),
                }
            )
            style_idx += 1

    df_style = pd.DataFrame(style_rows, columns=["id", "concept", "prompt"])
    style_output_path = "style_painting_prompts.csv"
    df_style.to_csv(style_output_path, index=False)

    print(f"Saved {len(df_style)} rows to {style_output_path}")
    
    CIFAR_rows = []
    cifar_idx = 1
    for concept in CIFAR100_classes:
        CIFAR_rows.append({
            "id": cifar_idx,
            "concept": concept,
        })
        cifar_idx += 1
    df_cifar = pd.DataFrame(CIFAR_rows, columns=["id", "concept"])
    cifar_output_path = "cifar100_concepts.csv"
    df_cifar.to_csv(cifar_output_path, index=False)
    print(f"Saved {len(df_cifar)} rows to {cifar_output_path}")