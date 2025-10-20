// MAIN RENDERING FUNCTION

function render(type, e) {//main rendering funcion

    if (type == 'click') isMouseDown = true;
    else if (type == 'up') isMouseDown = false;

    if (type == 'down' && !isMouseDown) return;


    if (e.which == 3 || type == 'fake') {
        if (type == 'click') e.preventDefault();
        type = 'alt';
    }

    let x = e.offsetX + translateCoords[0] || 0;
    let y = e.offsetY + translateCoords[1] || 0;


//->
    if (MODE == 'Move' && type != 'alt') translateRender(type, e, x, y);

    if (CONNECTION_SEL_MODE == "HighlightCircles") circles = circle.calculate();

    //drawing connection lines

    if (lastClickedElement != null && type == "down" && MODE == "Place") {
        if (e.altKey) { //moving a node
            /*if(getGraphNodeDistance(x,y)==false ||
               getGraphNodeDistance(x,y)==lastClickedElement)// keeping distance not implemented
              {*/
            prex = {x: lastClickedElement.name.x, y: lastClickedElement.name.y}
            lastClickedElement.name.x = x;
            lastClickedElement.name.y = y;
            prex = {x: prex.x - x, y: prex.y - y};

            // moving multiple nodes :
            ms.selected.forEach(e => {
                if (e != lastClickedElement.id) {
                    graph.nodes.get(e).name.x -= prex.x;
                    graph.nodes.get(e).name.y -= prex.y;
                }

            })
            //}

        } else { //  drawing connection line
            ctxo.clearAll();
            ctxo.beginPath();
            ctxo.moveTo(x, y);
            ctxo.lineTo(lastClickedElement.name.x, lastClickedElement.name.y);
            ctxo.stroke();
            drawingLine = true;

        }

    }
    //linking up eventinos

    //checking link locations
    //clickArea = false; //meaning -> we can draw
    clickArea = getGraphNodeDistance(x, y);
    updateShortcutKeys(type);


    //console.log(clickArea);


    if (type == 'click' && clickArea == false && MODE == 'Place' && e.shiftKey == false) {
        graph.addNode(new Node(graph.nodes.size, {x: x, y: y, text: null}));
        changeStruct();
        should.nodes = true;
    }


    ctx.clearAll();
    //rendering everything out
    if (clickArea.type == 'Node' && type == 'up' && drawingLine && MODE == "Place") {
        // connecting two things

        if (clickArea != lastClickedElement) {
            // toggling
            if (!graph.nodes.get(lastClickedElement.id).edges.has(clickArea.id)) {
                graph.nodes.get(lastClickedElement.id).edges.set(clickArea.id, 10);
                graph.nodes.get(clickArea.id).edges.set(lastClickedElement.id, 10);
                changeStruct();
                should.connections = true;
            }
            sidebar.showLine(lastClickedElement.id, clickArea.id);

            if (e.ctrlKey) sidebar.removeCurrLine();
            //changeStruct();
            graph.validate(true /*enforcing*/);
        }
        //else if (confirm('Do you want to connetct this node with itself?')){
        //   graph.nodes.get( lastClickedElement.id ).edges.set( clickArea.id , 10);
        //}
    }

    if (type == 'up') {
        lastClickedElement = null;
        ctxo.clearAll();
        drawingLine = false;
        if (e.ctrlKey && clickArea.type != undefined) {
            // -------- MULTI SELECT MANAGEMENT ------
            //ctrl is pressed
            //console.log('>>');
            ms.update();
            if (!ms.selected.has(clickArea.id)) ms.selected.add(clickArea.id)
            else ms.selected.delete(clickArea.id)
        } else if (!e.ctrlKey && !e.shiftKey) {
            ms.flush();
            ms.update();
        }
    }


    if (MODE == "Search" && lastClickedElement != null && clickArea != false && clickArea != true) {
        ConnectionHighlight = graph.dijkstra(clickArea, lastClickedElement);
    }

    // HOLY DRAWING THING
    ctxbg.clearAll();
    ctxbg.fillStyle = canvBgColor;
    ctxbg.fillRect(0, 0, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);


    graph.nodes.forEach((val, key) => {

        ctx.beginPath();
        ctx.lineWidth = 1;

        ctx.strokeStyle = "black";
        //NODE COLOR THINGS
        mainColor = renderSelectedColor(val);

        if (FORCE_LSA) {
            mainColor = LSA_xdiff[val.id]
        } else if (val.id === 0) { // highlight node
            ctx.lineWidth = 6;
            ctx.strokeStyle = MAT_COLORS['orange-800'];
        } else if (ms.selected.has(val.id)) {//element is selected
            ctx.lineWidth = 6;
            ctx.strokeStyle = MAT_COLORS['orange-800'];
            ctx.fillStyle = '#eeeeee';
        } else if (val == clickArea && type != 'up') {//clicking on elementt
            ctx.fillStyle = MAT_COLORS['blue-800'];
            sidebar.showEl(val, '#properties');
            ctx.lineWidth = 1;
            ctx.stokeStyle = "black";
        } else if (val == clickArea && type == 'up') {
            ctx.lineWidth = 6;
            ctx.strokeStyle = MAT_COLORS['blue-800'];
            ctx.fillStyle = '#eeeeee';
        } else if (val == lastClickedElement) {
            ctx.lineWidth = 6;
            ctx.strokeStyle = MAT_COLORS['green-500'];
            ctx.fillStyle = '#eeeeee';
        } else {
            ctx.fillStyle = mainColor;
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1;
        }

        if (val._rest.Type) {
            if (val._rest.Type === "cd") {
                ctx.fillStyle = MAT_COLORS['green-500'];

            }
            if (val._rest.Type === "cp") {

                ctx.fillStyle = MAT_COLORS['blue-500'];
            }
            if (val._rest.Type === "f") {
                ctx.fillStyle = MAT_COLORS['brown-400'];
            }
            if (val._rest.Type === "d") {
                ctx.fillStyle = MAT_COLORS['orange-600'];
                ctx.strokeStyle = MAT_COLORS['orange-500'];
                ctx.lineWidth = 10;
            }
        }

        if (e.shiftKey && SELECTING != false && SELECT_TO != false) {
            if (isInRectangle(...SELECTING, x, y, val.name.x, val.name.y)) {
                ctx.lineWidth = 3;

                ctx.strokeStyle = '#ccccccc';

                if (type == "up") ms.selected.add(val.id);
                ctx.strokeStyle = MAT_COLORS['orange-800'];
                ctx.fillStyle = '#eeeeee';
            }
        }

        if (val == clickArea && e.altKey && type != 'up') ctx.fillStyle = '#4488FF';

        let cRadius = (RADIUS / 10) * val.weight;

        if (NODE_WEIGHT_MODE == 'ConnectionWeight') {
            //calculating  manually from incoming connections.
            cRadius = (RADIUS / 10);
            val.edges.forEach((edge) => cRadius += (edge / 5));

        }

        ctx.arc(val.name.x, val.name.y, cRadius, 0, 2 * Math.PI);
        ctx.fill();


        ctx.fillStyle = fontColor.value || "#222";
        let fontSize = sidebar.fontSize || 12
        ctx.font = fontSize + 'px sans-serif';
        ctx.textAlign = 'center';

        // text manipulation and defaults
        let parsedText = '';

        if (val.name.text == null) {
            parsedText = parseInfo(textExtra[1] + textExtra[0] + textExtra[2], val);
        } else {
            parsedText = parseInfo(textExtra[1] + val.name.text + textExtra[2], val);
        }
        ctx.fillText(parsedText, val.name.x, val.name.y + val.weight * 2.5);


        ctx.stroke();


        if (val.treeRoot) {
            if (circle.subGraphHas.has(val.subgraph)) {
                // it was a root once but the container is no longer a tree so removing flag
                delete val.treeRoot;
            } else {
                ctx.beginPath();

                ctx.fillStyle = "#222";
                ctx.arc(val.name.x, val.name.y, (cRadius / 2.5), 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }

        }

        val.edges.forEach((w, edgeId) => {

            connectedEdge = graph.nodes.get(edgeId);
            //console.log(connectedEdge,edgeId);
            ctxbg.beginPath();
            ctxbg.strokeStyle = "transparent";
            ctxbg.moveTo(val.name.x, val.name.y);

            //Drawing connection weight out
            if (CONNECTION_MODE == "Manual") {
                ctxbg.lineWidth = (val.edges.get(edgeId) + 1) / 5;
            } else {
                dst = distance(val.name.x, connectedEdge.name.x, val.name.y, connectedEdge.name.y, true);

                let temp = 10 + (30 - Math.sqrt(dst) * 5) / 5;
                temp = (temp < 4) ? 3 : temp;
                val.edges.set(edgeId, temp);
                ctxbg.lineWidth = temp;
                //changeStruct();
            }

            // ---------------------------
            // Drawing connection color out

            if (ConnectionHighlight.has(edgeId) &&
                ConnectionHighlight.has(val.id) &&
                Math.abs(ConnectionHighlight.indexOf(edgeId) - ConnectionHighlight.indexOf(val.id)) == 1) {
                ctx.lineWidth = 1;
                ctx.fillStyle = MAT_COLORS['orange-500'];
                ctx.fill();
                ctx.arc(val.name.x, val.name.y, 10, 0, 2 * Math.PI);

                ctxbg.lineWidth = 6; //currSelNode
                ctxbg.strokeStyle = MAT_COLORS['orange-500'];
            } else if (val == clickArea) {
                ctxbg.setLineDash([10, 8]);
                ctxbg.lineWidth = 6; //currSelNode
                ctxbg.strokeStyle = MAT_COLORS['blue-500'];


            } else {
                if (CONNECTION_SEL_MODE == 'Default') ctxbg.strokeStyle = "#222222";
                else if (CONNECTION_SEL_MODE == 'FromWeight') {
                    let weight = val.edges.get(edgeId);
                    let parse16 = Math.round((weight / 30) * 255);
                    let parse = (parse16 > 255 ? 255 : parse16).toString(16); //converting width to HEX
                    ctxbg.strokeStyle = ("#222222" + parse);

                } else if (CONNECTION_SEL_MODE = "HighlightCircles") {
                    //See calculation code in circle.js

                    //checking whether the current pair is in the circle detection val.id vs edgeId
                    isInCircles = circles.filter(x => ((x[0] == val.id && x[1] == edgeId) || (x[1] == val.id && x[0] == edgeId)));

                    if (isInCircles.length > 0) {
                        ctxbg.strokeStyle = MAT_COLORS['green-500'];
                    } else {
                        ctxbg.strokeStyle = ("#444444");
                    }

                }

            }


            let weight = val.edges.get(edgeId);
            if (weight.includes("COLOR:")) {
                const col = weight.split(':')[1];
                const tour = col % MAT_COLORS.swatches.length;
                ctxbg.lineWidth = 2;
                console.log(MAT_COLORS.swatches[tour],weight)
                ctxbg.strokeStyle = MAT_COLORS[MAT_COLORS.swatches[tour] + '-500'];
            }


            // TREE RELATED FUNCTIONS


            ctxbg.lineTo(connectedEdge.name.x, connectedEdge.name.y);
            //later adding width (weight)
            ctxbg.stroke();
            ctxbg.setLineDash([]);

        });

    });


// in: X,Y coordinates
// @returns {ret} Boolean|Node
// - true if node in blocking range
// - NodeObject if clicked at node
// - false[default] if not blocked
    function getGraphNodeDistance(x, y) {
        let ret = false;
        graph.nodes.forEach((val, key) => {
            let currX = val.name.x;
            let currY = val.name.y;
            if (distance(currX, x, currY, y, val.weight) == -1) /*pithagoras*/
            {
                //we found a node

                if (type == 'click') lastClickedElement = val;
                ret = val;

            } else if (distance(currX, x, currY, y, RADIUS) == 0) { //pithagorash
                //we found a node but not in range
                ret = true;
            }
        });

        return ret;
    }

    ms.update();//update the render procedure

    if (e.shiftKey && type == 'click') SELECTING = [x, y];
    else if (e.shiftKey && type == 'down' & SELECTING != false) {
        drawSelRect(SELECTING[0], SELECTING[1], x - SELECTING[0], y - SELECTING[1]);
        //  ms.selected = new Set();
    } else [SELECTING, SELECT_TO] = [false, false];


}//render

function getGNDistance(x, y) {
    let ret = false;
    graph.nodes.forEach((val, key) => {
        let currX = val.name.x;
        let currY = val.name.y;
        if (distance(currX, x, currY, y, val.weight) == -1) /*pithagoras*/
        {
            //we found a node
            if (type == 'click') lastClickedElement = val;
            ret = val;
        } else if (distance(currX, x, currY, y, RADIUS) == 0 && ret == false) { //pithagorash
            //we found a node but not in range
            ret = true;
        }
    });
    return ret;
}

function getClosestNodes(nodeID, to = 1) {
    let index = [];
    x = graph.nodes.get(nodeID).name.x;
    y = graph.nodes.get(nodeID).name.y;

    graph.nodes.forEach((val, key) => {
        if (val.id != nodeID) {
            let currX = val.name.x;
            let currY = val.name.y;
            let dst = distance(currX, x, currY, y, true);
            index.push([dst, val.id]);
        }
    });
    sorted = index.sort((a, b) => a[0] < b[0] ? -1 : 1).splice(0, to);

    return {
        full: sorted,
        data: sorted.map(x => x[1]),
        dist: sorted.map(x => x[0])
    }


}

Array.prototype.random = function () {
    return this[Math.randInt(this.length)]
}


function drawSelRect(x0, y0, x1, y1) { //drawing selection rectangle x1 y1 are actually h and w
    ctxo.clearAll();

    let d = 10;
    //drawing main dotted rect
    ctxo.beginPath();
    ctxo.setLineDash([7, 7]);
    ctxo.lineWidth = 2;
    ctxo.strokeStyle = "#222";
    ctxo.fillStyle = "#EEEEEE99";
    ctxo.rect(x0, y0, x1, y1);
    ctxo.fill();
    ctxo.stroke();

    //resetting path
    ctxo.beginPath();
    ctxo.lineWidth = 2;
    ctxo.setLineDash([0, 0]);
    ctxo.fillStyle = "#222";

    //drawing small border rects
    ctxo.rect(x0, y0, d, d);
    ctxo.rect(x0 + x1 - d, y0 + y1 - d, d, d);
    ctxo.rect(x0, y0 + y1 - d, d, d);
    ctxo.rect(x0 + x1 - d, y0, d, d);
    ctxo.fill();
    ctxo.stroke();

    SELECT_TO = [x1, y1];
}

function isInRectangle(x0, y0, x1, y1, px, py) {
    // checking whether a point is in a rectangle
    let x = [x0, x1].nsort();
    let y = [y0, y1].nsort();

    if (px >= x[0] && py >= y[0] && py <= y[1] && px <= x[1]) return true;
    else return false;
}

Array.prototype.nsort = function () {
    return this.sort((a, b) => Number(b) < Number(a) ? 1 : -1)
};
