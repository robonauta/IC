import resources.*;
import java.io.*;
import javax.imageio.ImageIO;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.awt.Color;

// 0 - black
// t_max - white

public class ImageTransf {
    private static int tmax;

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

    private static int findTmax(int[][] binImg, int[][] SE, int i, int j) {
        int insideTmax = -1;
        int originX = SE.length / 2;
        int originY = SE[0].length / 2;
        for (int p = 0; p < SE.length; p++) {
            for (int q = 0; q < SE[0].length; q++) {
                if (SE[p][q] == 1) {
                    int distX = p - originX;
                    int distY = q - originY;
                    if ((i + distX) < 0 || (i + distX) >= binImg.length || (j + distY) < 0
                            || (j + distY >= binImg[0].length)) {
                        return tmax;
                    } else {
                        if (binImg[i + distX][j + distY] > insideTmax)
                            insideTmax = binImg[i + distX][j + distY];
                    }
                }
            }
        }
        return insideTmax;
    }

    private static int findTmin(int[][] binImg, int[][] SE, int i, int j) {
        int insideTmin = Integer.MAX_VALUE;
        int originX = SE.length / 2;
        int originY = SE[0].length / 2;
        for (int p = 0; p < SE.length; p++) {
            for (int q = 0; q < SE[0].length; q++) {
                if (SE[p][q] == 1) {
                    int distX = p - originX;
                    int distY = q - originY;
                    if ((i + distX) >= 0 && (i + distX) < binImg.length && (j + distY) >= 0
                            && (j + distY) < binImg[0].length) {
                        if (binImg[i + distX][j + distY] < insideTmin)
                            insideTmin = binImg[i + distX][j + distY];
                    }
                }
            }
        }
        return insideTmin;
    }

    private static int[][] operation(int[][] binImg, int[][] SE, boolean isDil) {
        int[][] result = new int[binImg.length][binImg[0].length];
        for (int i = 0; i < binImg.length; i++)
            for (int j = 0; j < binImg[0].length; j++)
                result[i][j] = binImg[i][j];

        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                if (isDil)
                    result[i][j] = findTmin(binImg, SE, i, j);
                else
                    result[i][j] = findTmax(binImg, SE, i, j);
            }

        }
        return result;
    }

    public static int[][] erode(int[][] img, int[][] SE) {
        return operation(img, SE, false);
    }

    public static int[][] dilate(int[][] img, int[][] SE) {
        return operation(img, SE, true);
    }

    public static int[][] open(int[][] img, int[][] SE) {
        return dilate(erode(img, SE), SE);
    }

    public static int[][] close(int[][] img, int[][] SE) {
        return erode(dilate(img, SE), SE);
    }

    private static void draw(int[][] binImg, String title) {
        int width = binImg.length;
        int height = binImg[0].length;
        Picture picture = new Picture(width, height);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                if (binImg[row][col] == 0)
                    picture.set(row, col, Color.BLACK);
                else
                    picture.set(row, col, new Color(255, 255, 255));

                // picture.set(row, col, new Color(binImg[row][col], binImg[row][col],
                // binImg[row][col]));
            }
        }
        picture.show(title);
    }

    private static void drawGray(int[][] binImg, String title) {
        int width = binImg.length;
        int height = binImg[0].length;
        Picture picture = new Picture(width, height);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                if (binImg[row][col] == 0)
                    picture.set(row, col, Color.BLACK);
                else
                    picture.set(row, col, new Color(binImg[row][col], binImg[row][col], binImg[row][col]));
            }
        }
        picture.show(title);
    }

    private static int[][] getGrayScale(Picture img) {
        int width = img.width();
        int height = img.height();
        int[][] grayImg = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color c = img.get(i, j);
                int r = c.getRed();
                int g = c.getGreen();
                int b = c.getBlue();
                int y = (int) Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                grayImg[i][j] = y;
                // Color graytone = new Color(y, y, y);
                // grayPic.set(i, j, graytone);
            }
        }
        return grayImg;
    }

    private static int[][] getBinary(Picture img) {
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
                    binImg[i][j] = 0;
                else
                    binImg[i][j] = 1;
            }
        }
        return binImg;
    }

    public static Picture toGrayScale(Picture img) {
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

    public static void main(String[] args) {
        Picture img = new Picture(args[args.length - 1]);
        In inputSE = new In(args[1]);
        int[][] imgMatrix;

        if (args[2].equals("-b")) {
            tmax = 1; // Image is binary
            imgMatrix = getBinary(img);
        } else if (args[2].equals("-g")) {
            tmax = 255; // Image is grayscale
            imgMatrix = getGrayScale(img);
        } else
            throw new IllegalArgumentException("Invalid option. -g for grayscale picture or -b for binary picture.");

        int m = inputSE.readInt();
        int n = inputSE.readInt();

        int[][] SE = new int[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                SE[i][j] = inputSE.readInt();

        img.show("Source image");
        if (tmax == 255)
            toGrayScale(img).show("Grayscale source image");

        switch (args[0]) {
        case "-d":
            StdOut.println("Dilating");
            int[][] dilated = dilate(imgMatrix, SE);
            if (tmax == 1)
                draw(dilated, "Dilated binary image");
            else
                drawGray(dilated, "Dilated grayscale image");
            break;
        case "-e":
            StdOut.println("Eroding");
            int[][] eroded = erode(imgMatrix, SE);
            if (tmax == 1)
                draw(eroded, "Eroded binary image");
            else
                drawGray(eroded, "Eroded grayscale image");
            break;
        case "-o":
            StdOut.println("Opening");
            int[][] opened = open(imgMatrix, SE);
            if (tmax == 1)
                draw(opened, "Opened image");
            else
                drawGray(opened, "Opened grayscale image");
            break;
        case "-c":
            StdOut.println("Closing");
            int[][] closed = close(imgMatrix, SE);
            if (tmax == 1)
                draw(closed, "Closed image");
            else
                drawGray(closed, "Closed grayscale image");
            break;
        default:
            throw new IllegalArgumentException(
                    "Invalid option. -d for dilate, -e for erode, -o for open and -c for close");
        }
    }
}