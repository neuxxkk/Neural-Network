import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.*;

float [][]positions = {
                      {-1,-1}, {0,-1}, {1,-1}, 
                      {-1,0}, {0,0}, {1,0},
                      {-1,1}, {0,1}, {1,1}
                   };
                   
PVector gridPos, lastPos;
ArrayList<Integer> array;
                   
final int pixSize = 10;

void setup(){
  size(280,330);
  background(255,255,255);
  frameRate(240);
  
  lastPos = new PVector(-1, -1);
  
  List<Integer> list = Collections.nCopies(784, 0);
  array = new ArrayList<Integer>(list);
}

void draw(){
  fill(150,150,150);
  noStroke();
  rect(0,0, width, 50);
  doneBtn();
  cleanBtn();
  
}

void cleanBtn(){
  //position
  int x = width - 70, y = 10;
  int larg = 60, alt = larg/2;
  
  //style
  stroke(0);
  fill(200,15,30);
  
  //draw
  rect(x, y, larg, alt);
}

void doneBtn(){
  //position
  int x = 10, y = 10;
  int larg = 60, alt = larg/2;
  
  //style
  stroke(0);
  fill(20,200,30);
  
  //draw
  rect(x, y, larg, alt);
  
  //update
  gridPos = screenToGrid(mouseX, mouseY);
}

PVector screenToGrid(int x, int y){
  return new PVector(constrain(int(x/10) * 10, 0, 270), constrain(int(y/10) * 10, 0, 320));
}

int gridToArrayPos(PVector gridPos){
  return constrain(((int((gridPos.y-50) / 10) * 28) + int(gridPos.x / 10)), 0, 783);
}

void mouseDragged(){
  if (!gridPos.equals(lastPos) && gridPos.y >= 50) drawPixel();
}

void mousePressed(){
  //position
  int xDone = 10, yDone = 10;
  int largDone = 60, altDone = largDone/2;
  
   //position
  int xClean = width - 70, yClean = 10;
  int largClean = 60, altClean = largClean/2;
  
  //Done
  if (mouseX > xDone && mouseX < xDone + largDone && mouseY > yDone && mouseY < yDone + altDone){
    cleanBtn();
    load();
    exit();
    
    //Clean
  }else if (mouseX > xClean && mouseX < xClean + largClean && mouseY > yClean && mouseY < yClean + altClean){
    background(255,255,255);
  }else if (gridPos.y >= 50){
    drawPixel();
  }
}

void drawPixel(){
  for (int i = 0; i < 9; i++) { // Criar pequenas variações
      
      float offsetX = positions[i][0] * pixSize; // Pequena variação horizontal
      float offsetY = positions[i][1] * pixSize; // Pequena variação vertical
      
      int alpha;
      
      if (i % 2 == 0 && i != 4) alpha = 30;
      else if (i == 4) alpha = 200;
      else alpha = 120;
      
      fill(0, alpha); 
      noStroke();
      rect(gridPos.x + offsetX, gridPos.y + offsetY, pixSize, pixSize); // Pequenos pontos simulando a tinta
      
      PVector nowGrid = new PVector(gridPos.x + offsetX, gridPos.y + offsetY);
      
      int idx = gridToArrayPos(nowGrid);
      int newValue = constrain(array.get(idx) + alpha, 0, 255);
      array.set(idx, newValue);
    }
    lastPos = gridPos;
}


void load(){

  String csv = array.stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(","));
                    
  try (BufferedWriter writer = new BufferedWriter(new FileWriter("/home/vitornms/Neural-Network/number.csv"))){
    writer.write(csv);
  }catch (IOException e){
    e.printStackTrace();
  }
}
