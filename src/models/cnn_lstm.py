import torch
import torch.nn as nn


# --------------------------------------------------
# CNN Encoder
# --------------------------------------------------
class CNNEncoder(nn.Module):

    def __init__(
        self,
        feature_dim=128,
        dropout=0.2
    ):

        super().__init__()

        # --------------------------------------------------
        # Block 1
        # NO early downsampling
        # preserve precise spatial localization
        # --------------------------------------------------

        self.block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU()
        )

        # --------------------------------------------------
        # Block 2
        # 128 -> 64
        # learned downsampling
        # --------------------------------------------------

        self.block2 = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU()
        )

        # --------------------------------------------------
        # Block 3
        # 64 -> 32
        # --------------------------------------------------

        self.block3 = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU()
        )

        # --------------------------------------------------
        # Block 4
        # 32 -> 16
        # --------------------------------------------------

        self.block4 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU()
        )

        # --------------------------------------------------
        # Adaptive pooling
        # Converts:
        #
        # (B,128,H,W)
        #
        # ->
        #
        # (B,128,1,1)
        # --------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------
        # Feature bottleneck
        # --------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(128, feature_dim),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        """
        x:
        (B,3,H,W)
        """

        x = self.block1(x)

        # (B,16,128,128)

        x = self.block2(x)

        # (B,32,64,64)

        x = self.block3(x)

        # (B,64,32,32)

        x = self.block4(x)

        # (B,128,16,16)

        x = self.pool(x)

        # (B,128,1,1)

        x = x.view(x.size(0), -1)

        # (B,128)

        feat = self.fc(x)

        # (B,feature_dim)

        return feat

# --------------------------------------------------
# CNN + LSTM Model
# --------------------------------------------------
class CNNLSTM(nn.Module):

    def __init__(
        self,
        feature_dim=128,
        hidden_dim=128,
        dropout=0.2
    ):

        super().__init__()

        # --------------------------------------------------
        # Visual encoder
        # --------------------------------------------------

        self.encoder = CNNEncoder(
            feature_dim=feature_dim,
            dropout=dropout
        )

        # --------------------------------------------------
        # Temporal dynamics
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # Coordinate prediction head
        # --------------------------------------------------

        self.fc_out = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 4)
        )

    # --------------------------------------------------
    # Initialize hidden state
    # --------------------------------------------------

    def init_hidden(
        self,
        batch_size,
        device
    ):

        h = torch.zeros(
            1,
            batch_size,
            self.lstm.hidden_size
        ).to(device)

        c = torch.zeros(
            1,
            batch_size,
            self.lstm.hidden_size
        ).to(device)

        return (h, c)

    # --------------------------------------------------
    # One autoregressive step
    # --------------------------------------------------

    def forward_step(
        self,
        img,
        hidden
    ):

        """
        img:
        (B,3,H,W)

        hidden:
        (h,c)

        Returns:
        pred_coords:
        (B,4)
        """

        # --------------------------------------------------
        # Visual feature extraction
        # --------------------------------------------------

        feat = self.encoder(img)

        # (B,feature_dim)

        # --------------------------------------------------
        # Add sequence dimension
        # --------------------------------------------------

        feat = feat.unsqueeze(1)

        # (B,1,feature_dim)

        # --------------------------------------------------
        # Temporal update
        # --------------------------------------------------

        out, hidden = self.lstm(
            feat,
            hidden
        )

        # out:
        # (B,1,hidden_dim)

        out = out.squeeze(1)

        # (B,hidden_dim)

        # --------------------------------------------------
        # Predict coordinates
        # --------------------------------------------------

        pred_coords = self.fc_out(out)

        # (B,4)

        return pred_coords, hidden

    # --------------------------------------------------
    # Full teacher-forced forward
    # --------------------------------------------------

    def forward_sequence(
        self,
        imgs,
        hidden=None
    ):

        """
        imgs:
        (B,T,3,H,W)

        Used for:
        - debugging
        - sanity checking
        - optional training modes
        """

        B, T, C, H, W = imgs.shape

        if hidden is None:

            hidden = self.init_hidden(
                B,
                imgs.device
            )

        features = []

        # --------------------------------------------------
        # Encode each frame
        # --------------------------------------------------

        for t in range(T):

            feat = self.encoder(
                imgs[:, t]
            )

            features.append(feat)

        features = torch.stack(
            features,
            dim=1
        )

        # (B,T,feature_dim)

        # --------------------------------------------------
        # Temporal modeling
        # --------------------------------------------------

        out, hidden = self.lstm(
            features,
            hidden
        )

        # (B,T,hidden_dim)

        preds = self.fc_out(out)

        # (B,T,4)

        return preds