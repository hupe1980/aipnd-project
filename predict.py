#!/usr/bin/env python3
import model_builder
import utils

def main():
    args = utils.parse_predict_args()
    
    device = utils.get_device(args.gpu)
    
    model = model_builder.load_model(args.filename)
    mapping = model_builder.get_labelmapping(args.mapping)
    
    ps, classes = model_builder.predict(args.image_path, model, 5, device)
    
    for p, cl in zip(ps, classes):
        print('{}: {:.2%}'.format(mapping[cl], p))
    
if __name__ == "__main__":
    main()