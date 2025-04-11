# PAULow
PAULow: Patch-based Attention U-Net for Low-resource learning

Segmentation is an essential tool for cell biologists and involves isolating cells or cellular features from microscopy images. An automated segmentation pipeline with high precision and accuracy can significantly reduce manual labor and subjectivity. Frequently, researchers would seek for a validated model available online and fine-tune it to meet their segmentation requirements. However, the established fine-tuning approach may involve online training or computationally intensive offline training. To address this, we propose an offline training pipeline requiring only tens of samples that are morphologically distinct from pre-training data. Specifically, we employed a patch-based attention U-Net trained with a threshold-based custom loss function.

-- ##Installation
