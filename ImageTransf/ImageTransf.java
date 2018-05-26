import resources.*;
import java.io.*;
import javax.imageio.ImageIO;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.awt.Color;

public class ImageTransf {
    private static final int DELAY = 1;

    public static int[][] invert(int[][] binImg) {
        int[][] result = new int[binImg.length][binImg[0].length];
        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                if (binImg[i][j] == 1)
                    result[i][j] = 0;
                else
                    result[i][j] = 1;
            }
        }
        return result;
    }

    public static int[][] dilate(int[][] binImg, int[][] SE) {
        int[][] result = new int[binImg.length][binImg[0].length];
        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                if (binImg[i][j] == 1) {
                    for (int p = 0; p < SE.length; p++) {
                        for (int q = 0; q < SE[0].length; q++) {
                            if (SE[p][q] == 1) {
                                int originX = SE.length / 2;
                                int originY = SE[0].length / 2;
                                int distX = p - originX;
                                int distY = q - originY;
                                if ((i + distX) > 0 && (i + distX) < binImg.length && (j + distY) > 0
                                        && (j + distY < binImg[0].length)) {
                                    result[i + distX][j + distY] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    public static int[][] erode(int[][] binImg, int[][] SE) {
        int[][] result = new int[binImg.length][binImg[0].length];
        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                result[i][j] = binImg[i][j];
            }
        }

        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                if (binImg[i][j] == 0) {
                    for (int p = 0; p < SE.length; p++) {
                        for (int q = 0; q < SE[0].length; q++) {
                            if (SE[p][q] == 1) {
                                int originX = SE.length / 2;
                                int originY = SE[0].length / 2;
                                int distX = p - originX;
                                int distY = q - originY;
                                if ((i + distX) > 0 && (i + distX) < binImg.length && (j + distY) > 0
                                        && (j + distY < binImg[0].length && binImg[i + distX][j + distY] == 1)) {
                                    result[i + distX][j + distY] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    public static int[][] open(int[][] binImg, int[][] SE) {
        return dilate(erode(binImg, SE), SE);
    }

    public static int[][] close(int[][] binImg, int[][] SE) {
        return erode(dilate(binImg, SE), SE);
    }

    public static void draw(int[][] binImg, String title) {
        int width = binImg.length;
        int height = binImg[0].length;
        Picture picture = new Picture(width, height);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                if (binImg[row][col] == 1)
                    picture.set(row, col, Color.BLACK);
                else
                    picture.set(row, col, Color.WHITE);
            }
        }
        picture.show(title);
    }

    public static Picture getGrayScale(Picture img) {
        Picture grayPic = new Picture(img.width(), img.height());
        int width = img.width();
        int height = img.height();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color c = img.get(i, j);
                int r = c.getRed();
                int g = c.getGreen();
                int b = c.getBlue();
                int y = (int) Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                Color graytone = new Color(y, y, y);
                grayPic.set(i, j, graytone);
            }
        }
        return grayPic;
    }

    public static int[][] getBinary(Picture img) {
        int width = img.width();
        int height = img.height();
        int[][] binImg = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color c = img.get(i, j);
                int r = c.getRed();
                int g = c.getGreen();
                int b = c.getBlue();
                if ((r >= 0 && r <= 100) && (g >= 0 && g <= 100) && (b >= 0 && b <= 100))
                    binImg[i][j] = 1;
                else
                    binImg[i][j] = 0;
            }
        }
        return binImg;
    }

    public static void main(String[] args) {
        Picture img = new Picture(args[args.length - 1]);
        In inputSE = new In(args[1]);

        int m = inputSE.readInt();
        int n = inputSE.readInt();

        int[][] SE = new int[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                SE[i][j] = inputSE.readInt();

        Picture imgG = getGrayScale(img);
        img.show("Source Image");

        int[][] binImg = getBinary(imgG);
        draw(binImg, "Binary image");

        switch (args[0]) {
        case "-d":
            StdOut.println("Dilating");
            int[][] dilated = dilate(binImg, SE);
            draw(dilated, "Dilated image");
            break;
        case "-e":
            StdOut.println("Eroding");
            int[][] eroded = erode(binImg, SE);
            draw(eroded, "Eroded image");
            break;
        case "-o":
            StdOut.println("Opening");
            int[][] opened = open(binImg, SE);
            draw(opened, "Opened image");
            break;
        case "-c":
            StdOut.println("Closing");
            int[][] closed = close(binImg, SE);
            draw(closed, "Closed image");
            break;
        default:
            StdOut.println("Invalid option. -d for dilate, -e for erode, -o for open and -c for close");
            break;
        }
    }
}