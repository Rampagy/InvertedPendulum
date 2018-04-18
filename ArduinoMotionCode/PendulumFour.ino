//**************************************
//*   AI Pendulum Control Platform Interface
//*   4/6/2018
//*   Cunningham
//***************************************

#include <avr/io.h>
#include <avr/interrupt.h>

//Hardware Assignments
#define  EncoderChannel_A  2
#define  EncoderChannel_B  7
#define  Step 5
#define  Dir  4

//Function prototypes
void  SetupTimer1(void);
void  SetupTimer0(void);
void  Encoder_ISR(void);
void  WriteStepSize(int Speed);
void  MotionCalculation(void);

//Create a structure for movement data
struct MovementParms
{
  signed int Position;
  signed int Velocity;
  signed int Acceleration;
  bool Direction;
};

//Create instance of three structures 
volatile struct MovementParms DesiredCart;
volatile struct MovementParms CurrentCart;
volatile struct MovementParms Encoder;

//Static Variables that are accessed in interrupts
volatile unsigned int abs_Current_Velocity = 0;
volatile unsigned int Counter = 0;
volatile unsigned int StepSpeed = 0;
volatile signed long timestep = 0;
volatile bool RunTask_10ms = 0;
bool echo = 0;


//**************************************
//  Interrupt used to generate the servo 
//  pulses
//  The lower the address the higher is the priority level
//  Vector Address 12 
//**************************************
ISR(TIMER1_COMPA_vect)  
{   
   if(StepSpeed < 64000)
   {
     //Check the direction pin and determine the position
     if(CurrentCart.Direction == 1)
     {
        digitalWrite(Dir,1);
        CurrentCart.Position++;
     }
     else
     {
       digitalWrite(Dir,0);
       CurrentCart.Position--;
     }
  
     //Step once
     digitalWrite(Step,1);
     digitalWrite(Step,0);
   }
   //Interrupt occurs when timer matches this value
   //Larger velocity means smaller OCR1A value
   //StepSpeed is calculated using another function and then stored 
   //in a global variable to be accessed by the interrupt routine
   OCR1A = StepSpeed;
}

//**************************************
//  10 ms Task Rate
//  Creates a periodic Task 
//  Vector Address 15
//**************************************
ISR(TIMER0_COMPA_vect)
{
  //Check to see if the flag was not cleared, this will track 
  //overruns in the 10ms task
  if(RunTask_10ms)
  {
    digitalWrite(LED_BUILTIN,1);
  }
  RunTask_10ms = 1;
}

//*******************************************
//  ISR routine for the Encoder Channel
//  
//  Routine creates and interrupt on the 
//  rising edge of pin 2. Then checks to 
//  see if the if the second channel is 
//  high or low to determine the direction 
//
// Vector Address 2
//******************************************
void Encoder_ISR()
{
  if(digitalRead(EncoderChannel_B))
  {
    Encoder.Position++;
  }
  else
  {
    Encoder.Position--;
  }
  //600 PPR encoder, need to count 0, so (-299 to 300) = 600
  if(Encoder.Position > 300)
  {
    Encoder.Position = -299;
  }
  else if(Encoder.Position < -299)
  {
    Encoder.Position = 300;
  }
  
}
//*******************************************
// WriteStepSize
//
// Pass in a velocity and update the step size
// global variable which is read by the interrupt
// to generate the stepper pulses. Speed is in 
// the units mm/min. 
//*******************************************
void WriteStepSize(signed int Speed)
{
  //Set the direction pin based on the sign of the speed
  if(Speed > 0)
  {
     CurrentCart.Direction = 1;
  }
  else
  {
     CurrentCart.Direction = 0;
  } 
  // Conversion Factor
  //  20000 mm      1 min      360 deg     1 rev       step          16,000 step
  // ----------- x -------- x -------- x -------- x ------------ =  ---------
  //     min        60 sec      rev        60 mm     0.125 deg       sec
  //
  //   16,000,000 cycles      1 sec            1,000 cycles
  //   ----------------- x --------------- = -----------
  //       1 sec             16,000 step          sec
  //  16,000,000/16,000   = 1,000 clock cycles  (20000 mm/min)
  //  16,000,000/8,000    = 2,000 clock cycles  (10000 mm/min)
  //  16,000,000/800      = 20,000 clock cycles (1000 mm/min)
  
  //For some reason the prescale set to zero causes problems, so 8 is the minimum
  //which means the clock is 16,000,000/8=2,000,000
  //  2,000,000/16,000   = 125 clock cycles  (20000 mm/min)
  //  2,000,000/8,000    = 250 clock cycles  (10000 mm/min)
  //  2,000,000/800      = 2500 clock cycles (1000 mm/min)

  //Check to prevent overflows, StepSpeed is 16 bit
  if(Speed > 500 || Speed < -500)
  {
     //This is derived from the table above using a Power trendline
     StepSpeed = (signed int)(2500000L/(abs(Speed)));
  }
  //This is like setting it to zero speed
  else
  {
     StepSpeed = 65000;
  }
}
//*******************************************
//  Configure Timer 1
//*******************************************

void SetupTimer1()
{
  //TCCRnA/B- Stands for Timer/Counter Control Registers.
  TCCR1A = 0;
  TCCR1B = 0;

  //TCNTn- Stands for Timer/Counter Register.
  TCNT1 = 0; // initialize the counter from 0

  //OCRnA/B- Stands for Output Compare Register.
  OCR1A = 65000; // sets the counter compare value

  //TIMSKn- Stands for Timer/Counter Mask In Registers.
  TCCR1B |= (1<<WGM12); // enable the CTC mode
  TCCR1B |= (1<<CS11); // 1/8 Prescale, 0 prescale was not working

  TIMSK1 |= (1<<OCIE1A); 
}

//*********************************************
//  Configure Timer 0
//*********************************************
void SetupTimer0()
{
  TCCR0A = 0; // set entire TCCR0A register to 0
  TCCR0B = 0; // same for TCCR0B
  TCNT0  = 0; //initialize counter value to 0
  // set compare match register for 2khz increments
  OCR0A = 155;// = (16*10^6) / (100*1024) - 1 = 155(must be <256)
  // turn on CTC mode
  TCCR0A |= (1 << WGM01);
  // 1024 prescaler - page 142 of datasheet
  TCCR0B |= (1 << CS00); 
  TCCR0B |= (1 << CS02);  
  // enable timer compare interrupt
  TIMSK0 |= (1 << OCIE0A);
}

//*****************************************
//  Main Setup Function
//
//
//*****************************************
void setup() 
{
  bool run_program = 0;
  //Set the pin modes
  pinMode(EncoderChannel_A,INPUT);
  pinMode(EncoderChannel_B,INPUT);
  pinMode(Step,OUTPUT);
  pinMode(Dir,OUTPUT);
  pinMode(3,INPUT);
  //Turn on the onboard LED
  pinMode(LED_BUILTIN,OUTPUT);
  digitalWrite(LED_BUILTIN,1);
  
  //Wait for the switch input to start things off
  while(true)
  {
    run_program = digitalRead(3);
    if(run_program == 1)
    {
       digitalWrite(LED_BUILTIN,0);
       break;
    }
  }
  //Attach an interrupt INT.0 (priority 2) to the Encoder pin to trigger on rising edge
  attachInterrupt(digitalPinToInterrupt(EncoderChannel_A), Encoder_ISR, RISING);
 
  //Disable interrupts
  cli();
  //Setup Timers
  SetupTimer1();
  SetupTimer0();
  //enable interrupts
  sei();
  Serial.begin(115200);
}


//********************************************
//  Main Loop
//********************************************
void loop() 
{
 
  if(RunTask_10ms)
  {
    //P Controller
    if((Encoder.Position > -4) && (Encoder.Position < 4))
    {
       DesiredCart.Velocity = 0;
    }
    else
    {
       DesiredCart.Velocity = (Encoder.Position)* 125;
    }
    
    if(DesiredCart.Velocity > 10000)
    {
       DesiredCart.Velocity = 10000;
    }
    else if (DesiredCart.Velocity < -10000)
    {
       DesiredCart.Velocity = -10000;
    }
    DesiredCart.Acceleration = 1000;
    
    //Call this function to compute the current velocity,this
    //takes into account the acceleration
    MotionCalculation();
  
    //Call function to covert the velocity from mm/min to timer value

    WriteStepSize(CurrentCart.Velocity);

    timestep++;
    Serial.print(",");
    Serial.print(timestep,DEC);
    Serial.print(",");
    Serial.print(Encoder.Position,DEC);
    Serial.print(",");
    Serial.print(Encoder.Velocity,DEC);
    Serial.print(",");
    Serial.print(CurrentCart.Position,DEC);
    Serial.print(",");
    Serial.print(CurrentCart.Velocity,DEC);
    Serial.print(",");
    Serial.print(CurrentCart.Acceleration,DEC);
    Serial.print(",");
    Serial.print(echo,DEC);
    Serial.print(",");
    Serial.println("");

    
    echo = Serial.read();
    //Clear the flag that gets set by the interrupt
    //This was added at end to detect overruns of the 10ms task
    RunTask_10ms = 0;
  }
}

//*********************************************
// MotionCalculation
//
// Uses all global structures
//*********************************************
void MotionCalculation(void)
{  
  //Moving in the positive direction
  if(CurrentCart.Velocity >= 0)
  
     //Determine if we are speeding up
     if(CurrentCart.Velocity < DesiredCart.Velocity)
     {
        //Check to see if we can add the acceleration without overshooting
        if(CurrentCart.Velocity < (DesiredCart.Velocity - DesiredCart.Acceleration))
        {
           CurrentCart.Velocity += DesiredCart.Acceleration;
        }
        else
        {
           CurrentCart.Velocity = DesiredCart.Velocity;
        }
     }
     //We are slowing down 
     else
     {
        //Check for overshoot
        if(CurrentCart.Velocity > (DesiredCart.Velocity + DesiredCart.Acceleration))
        {
           CurrentCart.Velocity -= DesiredCart.Acceleration;
        }
        else
        {
           CurrentCart.Velocity = DesiredCart.Velocity;
        }
     }
  //Moving in the negative direction, all the signs flip   
  else
  {
     //Determine if we are speeding up
     if(CurrentCart.Velocity > DesiredCart.Velocity)
     {
        //Calculate the timer for the next pulse
        if(CurrentCart.Velocity > (DesiredCart.Velocity + DesiredCart.Acceleration))
        {
           CurrentCart.Velocity -= DesiredCart.Acceleration;
        }
        else
        {
           CurrentCart.Velocity = DesiredCart.Velocity;
        }
     }
     //We are slowing down 
     else
     {
        //Calculate the timer for the next pulse
        if(CurrentCart.Velocity < (DesiredCart.Velocity - DesiredCart.Acceleration))
        {
           CurrentCart.Velocity += DesiredCart.Acceleration;
        }
        else
        {
           CurrentCart.Velocity = DesiredCart.Velocity;
        }
     }
  }
}

