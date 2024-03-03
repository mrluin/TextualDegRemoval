# Datasets Preparation

----

#### Train image-to-text mapping (i2t mapper)

To enable i2t mapper can encode both clean and degraded concepts into textual word embedding space, we adopt both 
high-quality and degraded data to train i2t mapper.

| Dataset      | download from |
|--------------|---------------|
| LSDIR        |               |
| HQ-50K       |               |
| GoPro        |               |
| DPDD         |               |
| Rain200H     |               |
| Rain200L     |               |
| DID-Data     |               |
| DDN-Data     |               |
| SOTS-indoor  |               |
| SOTS-outdoor |               |

Please download and save the above datasets as follows:

```
dataroot | hq     | LSDIR    | xxxx.png 
         |        | HQ-50K   | xxxx.png
         | deblur | GoPro    | LQ            | xxxx.png
         |        |          | HQ            | xxxx.png
         |        | DPDD     | LQ            | xxxx.png
         |                   | HQ            | xxxx.png
         | derain | Rain200L | LQ            | xxxx.png
         |        |          | HQ            | xxxx.png
         |        | Rain200H | LQ            | xxxx.png
         |        |          | HQ            | xxxx.png
         |        | DID-Data | LQ            | xxxx.png
         |        |          | HQ            | xxxx.png
         |        | DDN-Data | LQ            | xxxx.png
         |                   | HQ            | xxxx.png
         | dehaze | indoor   | LQ            | xxxx.png
                  |          | HQ            | xxxx.png
                  |          | meta_info.txt  
                  | outdoor  | LQ            | xxxx.png
                             | HQ            | xxxx.png
                             | meta_info.txt
```

#### Train text restoration mapping (tr mapper)

In the second stage, we adopt pairs of LQ-HQ data to train out tr mapper.

