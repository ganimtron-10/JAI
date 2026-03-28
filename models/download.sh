#!/bin/bash

BASE_MODEL_DIR="./models"
mkdir -p "$BASE_MODEL_DIR"

MODELS=(
    "SmolLM2-135M-Instruct  | HuggingFaceTB/SmolLM2-135M-Instruct | smollm2-135m-instruct"
    "Qwen2.5-0.5B-Instruct  | Qwen/Qwen2.5-0.5B-Instruct          | qwen2.5-0.5b-instruct"
    "TinyLlama-1.1B-Chat    | TinyLlama/TinyLlama-1.1B-Chat-v1.0  | tinyllama-1.1b-chat"
)

FILES=(
    "model.safetensors"
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
)

echo "---------------------------"
echo "  JAI : Model Downloader"
echo "---------------------------"

select opt in "${MODELS[@]%%|*}" "Quit"; do
    if [[ "$REPLY" == "q" || "$opt" == "Quit" ]]; then
        echo "Exiting..."
        exit 0
    elif [ -n "$opt" ]; then
        SELECTED_DATA=${MODELS[$((REPLY-1))]}
        REPO_ID=$(echo "$SELECTED_DATA" | cut -d'|' -f2 | xargs)
        FOLDER_NAME=$(echo "$SELECTED_DATA" | cut -d'|' -f3 | xargs)
        TARGET_DIR="$BASE_MODEL_DIR/$FOLDER_NAME"
        
        mkdir -p "$TARGET_DIR"

        for FILE in "${FILES[@]}"; do
            URL="https://huggingface.co/$REPO_ID/resolve/main/$FILE"
            TARGET_FILE="$TARGET_DIR/$FILE"

            # SAFEGUARD: Check if file exists and has size > 0
            if [ -s "$TARGET_FILE" ]; then
                echo "⏭️  $FILE already exists. Skipping..."
            else
                echo "📥 Downloading $FILE..."
                # -C - allows curl to resume interrupted downloads
                curl -L -f "$URL" -o "$TARGET_FILE" -C -
                
                if [ $? -ne 0 ]; then
                    echo "⚠️  Note: $FILE not found or skipped."
                fi
            fi
        done

        echo -e "\n✅ Check complete for $TARGET_DIR"
        break
    else
        echo "❌ Invalid choice."
    fi
done