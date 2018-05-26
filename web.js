function GetBinary() {
    imgElement = document.getElementById('myImage');
    var canvas = document.createElement("canvas");
    canvas.width = imgElement.offsetWidth;
    canvas.height = imgElement.offsetHeight;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(imgElement, 0, 0);

    var map = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var imdata = map.data;

    var r, g, b;

    var currentInnerArray;
    var zeroesAndOnes = [];
    for (var p = 0, len = imdata.length; p < len; p += 4) {
        r = imdata[p]
        g = imdata[p + 1];
        b = imdata[p + 2];

        if (p % canvas.width * 4 === 0) {
            currentInnerArray = [];
            zeroesAndOnes.push(currentInnerArray);
        }
        if ((r >= 164 && r <= 255) && (g >= 191 && g <= 229) && (b >= 220 && b <= 255)) {
            currentInnerArray.push(0);
            // black  = water
            imdata[p] = 0;
            imdata[p + 1] = 0;
            imdata[p + 2] = 0;

        } else {
            currentInnerArray.push(1);
            // white = land
            imdata[p] = 255;
            imdata[p + 1] = 255;
            imdata[p + 2] = 255;
        }
    }
    ctx.putImageData(map, 0, 0);
    imgElement.src = canvas.toDataURL();
    console.log(zeroesAndOnes[1][1]);
    console.log("i : " + zeroesAndOnes.length + " j: " + zeroesAndOnes[1].length );
    return zeroesAndOnes;
}				