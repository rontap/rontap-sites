
// rendering a single circle out
function renderCircle(circle,text) {
    if (circle.isAdvanced == false && circle.isShowing == false) circle.isMoving = false
    if (circle.isShowing == false) return false;

    ctx.lineWidth=1
    ctx.beginPath()
    ctx.arc(circle.x, circle.y, circle.radius, 0, 2 * Math.PI)
    ctx.fillStyle= circle.color
    ctx.strokeStyle=borderColor
    ctx.fill()
    ctx.stroke()

    if (circle.isAdvanced && circle.advancedTimeLeft>=0) {
        if (circle.isMoving && circle.isAdvanced == 'collider' ) circle.advancedTimeLeft --;
        if (circle.isMoving && circle.isAdvanced == 'spawner' ) circle.advancedTimeLeft -=1.2;
        if (circle.isMoving && circle.isAdvanced == 'runner' ) circle.advancedTimeLeft -=1.25;
        if (circle.isMoving && circle.isAdvanced == 'killer' ) circle.advancedTimeLeft -=.3;

        if (circle.isAdvanced == 'killer') ctx.lineWidth=15
        else if (circle.isAdvanced == 'negator') ctx.lineWidth=2
        else  ctx.lineWidth=8
       
        
        ctx.beginPath()
        if (circle.isAdvanced == 'halfling') ctx.setLineDash([3,6])
        else if (circle.isAdvanced == 'vanisher') ctx.setLineDash([10,10])
        else ctx.setLineDash([])
        ctx.strokeStyle=Circle.advs[circle.isAdvanced].color;
        ctx.arc(circle.x, circle.y, circle.radius-5, -Math.PI/2, (2 * Math.PI )*((circle.advancedTimeLeft) / advancedTimeCoefficient)-Math.PI/2 )   
        
        ctx.stroke()
        if ( circle.advancedTimeLeft < 0 && circle.isAdvanced == 'collider') { 
            circle.isAdvanced = false;
            circle.collided += circle.advancedClicksLeft*2 + Math.floor(Math.random()*10 )
            }
        else if  ( circle.advancedTimeLeft < 0 && circle.isAdvanced == 'spawner') {
            circle.advancedTimeLeft = advancedTimeCoefficient;
            circles.push(new AdvCircle('halfling'));
        } 
        else if  ( circle.advancedTimeLeft < 0 && circle.isAdvanced == 'blocker') {
            circle.advancedTimeLeft = advancedTimeCoefficient;
            
        } 
        else if  ( circle.advancedTimeLeft < 0 && circle.isAdvanced == 'runner') {
            circle.advancedTimeLeft = advancedTimeCoefficient;
            if ( Math.abs(circle.deltaX) < 3 && Math.abs(circle.deltaY) < 3) {
                circle.deltaX *= 1.2;
                circle.deltaY *= 1.2;
            }
          
        } 
        
        else if  ( circle.advancedTimeLeft < 0 && circle.isAdvanced == 'killer') {
            alert('Game Over -> Killer on the loose');
        }
        
        
        if  ( circle.advancedClicksLeft <= 0 && circle.isAdvanced == 'halfling') {
            circle.isMoving = false;
            circle.isShowing = false;
            circle.radius=0;
        }
        if ( circle.advancedClicksLeft <= 0 && circle.isAdvanced == 'negator') {
            circle.advancedClicksLeft=1;
           addAdvCircle('halfling')
           
        }
        if (circle.isAdvanced == 'negator') {
            ctx.lineWidth=2
            ctx.beginPath()
            ctx.arc(circle.x, circle.y, circle.radius-10, 0, 2 * Math.PI)
            ctx.stroke()
            ctx.beginPath()
            ctx.arc(circle.x, circle.y, circle.radius-15, 0, 2 * Math.PI)
            ctx.stroke()
            ctx.beginPath()
            ctx.arc(circle.x, circle.y, circle.radius-20, 0, 2 * Math.PI)
            ctx.stroke()
        }
        if (circle.advancedClicksLeft <= 0) { circle.isAdvanced = false;  circle.isMoving = false;}
    }
    // added functionality: the order of the circles is shown with text
    ctx.setLineDash([])
    ctx.beginPath()
    ctx.textAlign = "center"
    ctx.textBaseline = "middle"
    ctx.fillStyle = textColor
    ctx.strokeStyle = textColor;
    ctx.font = "30px Courier New"
    
    if (circle.isAdvanced ) {
        if (circle.isAdvanced =='negator')  ctx.fillText('×',circle.x,circle.y)
        else ctx.fillText('+'+circle.advancedClicksLeft,circle.x,circle.y)
        
    }
    else if ((!tutorialMode || tutorialStage!=0) && (circle.isMoving)) {
        ctx.fillText(circle.collided, circle.x, circle.y)
    }
    
    ctx.stroke()
}


// main render loop. 
// in this implementation, the render and the tickspeed are linked,
// however, the maximum speed can be changed via the 'speed' variable
ping = [];
tick = 0;
level = 1;

shouldDrawLines = true;

function loop() {
    let ts = performance.now();
    tick++;
    ctx.clearRect(0,0,innerWidth,innerHeight)
    for (let i=0; i< circles.length; i++) {
        //looping and rendering each circle out
        renderCircle(circles[i],i);
        circles[i].checkEdge();
        circles[i].iterate();
    }

    sumCollisions = circles.reduce( (acc,el) => acc + el.collided , 0 )
    maxCollisions = circles.reduce( (acc,el) => (acc > el.collided) ? acc : el.collided , 0)
    timeSpentLevel= new Date().getTime() - timeSpent;
    totalCollisions.innerHTML = sumCollisions;
    maximumCollisions.innerHTML = maxCollisions;
    timeSpentOnLevel.innerHTML = Math.floor(timeSpentLevel/100)/10 

    // AUTOMATICALLY ADDING ELEMENTS 
    if (!tutorialMode) {
        if (goalCircles > circles.length && tick%10==0) {
            if      (circles.length % 6 == 0)   addAdvCircle('collider');
            else if (circles.length % 20 == 0)  addAdvCircle('killer');
            else if (circles.length % 9 == 0)   addAdvCircle('spawner');
            else if (circles.length % 13 == 0)  addAdvCircle('blocker');
            else if (circles.length % 7 == 0)   addAdvCircle('runner');
            else if (circles.length % 15 == 0)  addMultiple(['halfling'].times(2))
            else if (circles.length % 17 == 0)  addAdvCircle('vanisher');
            else     addCircle(false);
        }
    }
   
        
    

    if (tick % 20 == 0) renderUI(false)

    if (isAllStopped() && circles.length > 2)  nextLevel(++level)   
    else if (!isGamePaused)  setTimeout( () => {
        ping.push( (performance.now() - ts) )
        ping = ping.slice(-10);
        let currFPS = tickSpeed - Math.avg( ping );
        debugPing.innerHTML = (currFPS>0) ? currFPS : 'lagging';
        loop()
    
    } , 1000 / tickSpeed )
}

powerupsExpanded = false
function renderUI(filled = false) {
    ctxUI.clearRect(0,0,innerWidth,innerHeight);

    
    ctxUI.beginPath()
    ctxUI.strokeStyle="#444444"
    ctxUI.fillStyle="#cccccc44"
    ctxUI.moveTo(0,innerHeight-80);
    if (powerupsExpanded) {
        powerups.classList.add('on');
        ctxUI.beginPath()
        ctxUI.arc(-30, innerHeight-40,40, Math.PI ,0,false)   
        ctxUI.arc(40, innerHeight-40,30, Math.PI ,0,true)
        ctxUI.arc(110, innerHeight-40,40, Math.PI ,1.5*Math.PI,false)
        ctxUI.lineTo(innerWidth,innerHeight-80);
        ctxUI.lineTo(innerWidth,innerHeight);
        ctxUI.lineTo(0,innerHeight);
       
        flash.innerHTML="close";
     
    }
    else {
        powerups.classList.remove('on');
       
        ctxUI.lineTo(200,innerHeight-80);
        //ctxUI.arc(250, innerHeight-40,40, -Math.PI/2, 0,false);
        //ctxUI.lineTo(290,innerHeight);
        ctxUI.lineTo(innerWidth,innerHeight-80);
        ctxUI.lineTo(innerWidth,innerHeight);
        ctxUI.lineTo(0,innerHeight);
        flash.innerHTML="flash_on";
    }
    
    ctxUI.fill();
    ctxUI.stroke();

   
    let fillPart = (filled) ? 1000 : 1000

    ctxUI.lineWidth=6
    ctxUI.lineCap="round"
        
    ctxUI.beginPath()
    ctxUI.strokeStyle="#FF9800"
    ctxUI.fillStyle="#FF9800"
    ctxUI.arc(40, innerHeight-40,20, -Math.PI/2, (2 * Math.PI )*(fillPart-250) / 1000)
    
   
    
    if (filled) ctxUI.fill()
    ctxUI.stroke()
    ctxUI.beginPath()
    ctxUI.lineWidth=2
    ctxUI.strokeStyle="#444444"
    if (!powerupsExpanded)  ctxUI.arc(40, innerHeight-40,30, 0 ,(2 * Math.PI ))
    
    ctxUI.stroke();

    (filled) ? flash.classList.add('on') : flash.classList.remove('on')
    let colors=['#f44336','#3F51B5','#4CAF50']
    let data=[.5,((timeSpentLevel/1000)/maxTime),(sumCollisions/sumColls),(maxCollisions/maxColls)]

    let uiDrawTimes = (tutorialMode ? (tutorialStage > 3 ? 4 : tutorialStage+1) : 4 )
    for (let i =1; i<4;i++) {
        
        ctxUI.beginPath()
        ctxUI.lineWidth=5
        ctxUI.strokeStyle=colors[i-1]
        ctxUI.arc(40+(i*70), innerHeight-40,20, -Math.PI/2, (2 * Math.PI )*((data[i]*1000-250) / 1000))    
        ctxUI.stroke()
        ctxUI.lineWidth=2
    }    
}
