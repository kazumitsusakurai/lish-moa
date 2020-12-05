# Mechanisms of Action (MoA) Prediction

26th place solution for [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/c/lish-moa)

# Reference

* Summary
* Final Submission

# Demo

## Making DAE features

```
bin/01-dae.sh
```

## Training

```
bin/02-running-nn.sh
bin/03-running-tabnet.sh
bin/04-running-deepinsignt.sh
```

These also can be used for the only prediction. (learned weights for a model are required)

```
bin/02-running-nn.sh --only-pred
bin/03-running-tabnet.sh --only-pred
bin/04-running-deepinsignt.sh --only-pred
```

# Summary

## Denoising Autoencoder
I used normal denoising autoencoder instead of the swapping method, and I concatenated it with the original features. It improved the public test socre from 0.01842 to 0.01834 but CV was not improved.

## Weighted Loss

I used weighted loss which weights targets that only occur few times in the train data (I used 40).
It significantly improved CV score (about -0.0003) but public test score wasn't improved. In order to overfit to the train data I blended both weighted-loss models and non-weighted-loss models.

## Blending

I adopted the simple weighted average of
  * 2 Hidden Layer NN (7 seed, 7 OOF, Train each model with weighted-loss and non-weighted-loss)
  * 3 Hidden Layer NN (same as the above)
  * 4 Hidden Layer NN (same as the above)
  * TabNet (6 seed)
  * DeepInsight Model (2 seed, different settings)

## Other Stuffs That Worked

* Rank gauss
* Adding statistical features (sum, mean, std, kurt, skew, median, etc..)
* PCA (only applying to TabNet)
* Smoothing loss

# Score

The Final submission score

- Private: 0.01608
- Public: 0.01820
- CV: 0.01550 (without ctl_vehicle)
