from calculate_prediction import predict
import argparse
import pathlib
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's identify the image!")
    parser.add_argument('--path', type=str, default=None,
                        help='Path of the image')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path of Saved Keras Model in .h5 file')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top K most likely classes')
    parser.add_argument('--category_names', type=pathlib.Path,
                        default=None,
                        help='Path to JSON file of all category names')
    
    args = parser.parse_args()
    probs, labels = predict(args.path, args.model_path, args.top_k)
    
    if args.top_k is None:
        print(f'Most likely image class is {labels[0]} '
              f'with associated probability {probs[0]}')
    else:
        print(f'Top {args.top_k} labels are: {labels}')
        print(f'Top {args.top_k} probabilities are: {probs}')
    
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
            
        topk_labels = [class_names[x] for x in labels]
        
        if args.top_k is None:
            print(f'And corresponding category name {topk_labels[0]}')
        else:
            print(f'And corresponding category names of top {args.top_k} '
            f'labels are: {topk_labels}')
        
