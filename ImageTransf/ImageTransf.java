
/******************************************************************************
 *  Compilation:  javac ImageTransf.java
 *  Execution:    java ImageTransf -co SE -g media/salt-and-pepper.png
 *  Dependencies: Picture.java In.java StdDraw.java StdOut.java
 *
 * 
 *  This program takes an operation, a structuring element, an image kind and
 *  an image as arguments and prints the output of that operation on the specified
 *  image. 
 *
 *  Structuring element syntax (only binary SEs): 
 *  number_of_rows number_of_columns
 *  draw of the specified SE as an matrix of 0's and 1's
 *  0 for black regions and 1 for white regions
 *
 *  Author: Pedro Henrique Barbosa de Almeida
 ******************************************************************************/

import resources.*;
import java.io.*;
import javax.imageio.ImageIO;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.awt.Color;

public class ImageTransf {
    private static int tmax = 255;
    private static boolean isBinary;
    private static boolean isColorful = false;

    // Invert colors of a binary image
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

    // Find maximum intensity value of a window
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

    // Find minimium intensity value of a window
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

    // Generic operation on a image
    private static int[][] operation(int[][] binImg, int[][] SE, boolean isDil) {
        int[][] result = new int[binImg.length][binImg[0].length];
        for (int i = 0; i < binImg.length; i++)
            for (int j = 0; j < binImg[0].length; j++)
                result[i][j] = binImg[i][j];

        for (int i = 0; i < binImg.length; i++) {
            for (int j = 0; j < binImg[0].length; j++) {
                if (!isDil)
                    result[i][j] = findTmin(binImg, SE, i, j);
                else
                    result[i][j] = findTmax(binImg, SE, i, j);
            }
        }
        return result;
    }

    public static int[][] gradient(int[][] img, int[][] SE) {
        int[][] delta = dilate(img, SE);
        int[][] epsilon = erode(img, SE);
        int[][] gradient = new int[img.length][img[0].length];
        for (int i = 0; i < img.length; i++)
            for (int j = 0; j < img[0].length; j++)
                gradient[i][j] = delta[i][j] - epsilon[i][j];
        return gradient;
    }

    // Erosion
    public static int[][] erode(int[][] img, int[][] SE) {
        return operation(img, SE, false);
    }

    // Dilation
    public static int[][] dilate(int[][] img, int[][] SE) {
        return operation(img, SE, true);
    }

    // Opening
    public static int[][] open(int[][] img, int[][] SE) {
        return dilate(erode(img, SE), SE);
    }

    // Closing
    public static int[][] close(int[][] img, int[][] SE) {
        return erode(dilate(img, SE), SE);
    }

    /*
     * // Draw a binary image private static void drawBin(int[][] binImg, String
     * title) { int width = binImg.length; int height = binImg[0].length; Picture
     * picture = new Picture(width, height); for (int row = 0; row < width; row++) {
     * for (int col = 0; col < height; col++) { if (binImg[row][col] == 0)
     * picture.set(row, col, Color.BLACK); else picture.set(row, col, new Color(255,
     * 255, 255)); // picture.set(row, col, new Color(binImg[row][col],
     * binImg[row][col], // binImg[row][col])); } } picture.show(title); }
     */

    // Draw a grayscale image
    private static void draw(int[][] img, String title) {
        int width = img.length;
        int height = img[0].length;
        Picture picture = new Picture(width, height);
        for (int row = 0; row < width; row++)
            for (int col = 0; col < height; col++)
                picture.set(row, col, new Color(img[row][col], img[row][col], img[row][col]));
        picture.show(title);
    }

    // Extract intensity from an grayscale image to a matrix
    private static int[][] getIntensity(Picture img) {
        SET<Integer> colorsOftheImage = new SET<Integer>();
        int width = img.width();
        int height = img.height();
        int[][] grayImg = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color c = img.get(i, j);
                int g = c.getGreen();
                if (c.getGreen() == c.getRed() && c.getBlue() == c.getGreen()) {
                    colorsOftheImage.add(g);
                    grayImg[i][j] = g;
                } else {
                    isColorful = true;
                    return getIntensity(toGrayScale(img));
                }
            }
        }
        if (colorsOftheImage.size() <= 2)
            isBinary = true;
        else
            isBinary = false;
        return grayImg;
    }

    /*
     * // Extract intensity from a binary image to a matrix private static int[][]
     * getBinary(Picture img) { int width = img.width(); int height = img.height();
     * int[][] binImg = new int[width][height]; for (int i = 0; i < width; i++) {
     * for (int j = 0; j < height; j++) { Color c = img.get(i, j); int r =
     * c.getRed(); int g = c.getGreen(); int b = c.getBlue(); if ((r == 0) && (g ==
     * 0) && (b == 0)) binImg[i][j] = 0; else binImg[i][j] = 1; } } return binImg; }
     */

    // Parse an image to grayscale
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
        StdOut.println("Type the letter corresponding to the operation you want: ");
        StdOut.println("d - dilation");
        StdOut.println("e - erosion");
        StdOut.println("o - opening");
        StdOut.println("c - closing");
        StdOut.println("co - close-open");
        StdOut.println("oc - open-close");
        StdOut.println("g - gradient");
        String op = StdIn.readString();
        StdOut.print("\nType the SE filename: ");
        String StructuringElement = StdIn.readString();
        StdOut.print("\nType the image filename:  ");
        String image = StdIn.readString();
        StdOut.println("\nRun in verborragic mode?: ");
        StdOut.println("y - yes");
        StdOut.println("n - no");
        String verb = StdIn.readString();

        Picture img = new Picture(image);
        In inputSE = new In(StructuringElement);
        int[][] imgMatrix;

        imgMatrix = getIntensity(img);

        int m = inputSE.readInt();
        int n = inputSE.readInt();

        int[][] SE = new int[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                SE[i][j] = inputSE.readInt();

        img.show("Source image");
        if (isColorful)
            toGrayScale(img).show("Grayscale source image");

        int[][] result;
        switch (op) {
        case "d":
            StdOut.println("Dilating");
            result = dilate(imgMatrix, SE);
            draw(result, "Dilated grayscale image");
            break;
        case "e":
            StdOut.println("Eroding");
            result = erode(imgMatrix, SE);
            draw(result, "Eroded grayscale image");
            break;
        case "o":
            StdOut.println("Opening");
            result = open(imgMatrix, SE);
            draw(result, "Opened grayscale image");
            break;
        case "c":
            StdOut.println("Closing");
            result = close(imgMatrix, SE);
            draw(result, "Closed grayscale image");
            break;
        case "oc":
            StdOut.println("Opening then closing");
            result = close(open(imgMatrix, SE), SE);
            draw(result, "Open-Closed grayscale image");
            break;
        case "co":
            StdOut.println("Closing then opening");
            result = open(close(imgMatrix, SE), SE);
            draw(result, "Close-opened grayscale image");
            break;
        case "g":
            StdOut.println("Gradient");
            result = gradient(imgMatrix, SE);
            draw(result, "Grayscale image gradient");
            break;
        default:
            throw new IllegalArgumentException(
                    "Invalid option. d for dilate, e for erode, o for open, c for close and g for gradient");
        }
        
        StdOut.println("isBinary?: " + isBinary);
        StdOut.println("isColorful?: " + isColorful);
        StdOut.println("isGray?: " + !(isColorful || isBinary));

        if (verb.equals("y")) {
            StdOut.println("Structuring element: ");
            for (int i = 0; i < SE.length; i++) {
                for (int j = 0; j < SE[0].length; j++)
                    StdOut.print(SE[i][j] + " ");
                StdOut.println();
            }
            StdOut.println("\nImage: ");
            if (isBinary) {
                for (int i = 0; i < imgMatrix.length; i++) {
                    for (int j = 0; j < imgMatrix[0].length; j++) {
                        if (imgMatrix[i][j] == 255)
                            StdOut.print("1 ");
                        else
                            StdOut.print(imgMatrix[i][j] + " ");
                    }
                    StdOut.println();
                }
            } else {
                for (int i = 0; i < imgMatrix.length; i++) {
                    for (int j = 0; j < imgMatrix[0].length; j++)
                        StdOut.print(imgMatrix[i][j] + " ");
                    StdOut.println();
                }
            }
            StdOut.println("\nOperated image: ");
            if (isBinary) {
                for (int i = 0; i < result.length; i++) {
                    for (int j = 0; j < result[0].length; j++) {
                        if (result[i][j] == 255)
                            StdOut.print("1 ");
                        else
                            StdOut.print(result[i][j] + " ");
                    }
                    StdOut.println();
                }
            } else {
                for (int i = 0; i < result.length; i++) {
                    for (int j = 0; j < result[0].length; j++)
                        StdOut.print(result[i][j] + " ");
                    StdOut.println();
                }
            }
        }
    }
}