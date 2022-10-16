#include<iostream>
#include<cstring>
#include<string>
#include<string.h>
#include<ctime>
#include<cstdlib>
#include<windows.h>
#include<iomanip>
using namespace std;

#define SIZE 20

string c_words[SIZE] = {"entrepreneur","rhythm","handkerchief","embarrass","millennium","pronunciation","deductible","acknowledgment",
                    "adequate","adjacency","distinguish","enthusiasm","australia","absorption","adolescent","align","analyze","anonymous",
                    "apparatus","arithmetic"};
string w_words[SIZE] = {"entreprenour","rythm","hankerchief","embarass","millinnium"," prononciation","deductable","acknowledegmnt",
                    "adaquate","adjacancy","distinguesh","enthusiasm","austrailia","absorbtion","adolecent","aline","analyse","annonomus",
                    "aparatus","arithmatic"};

void gotoxy(int,int);
void Above_horizental();
void Line();
void below_Horizental();

class Fighter
{
private:
    string name;
    int health;
    int points;
public:
    Fighter():name(""),health(0),points(0){}
    Fighter(string n,int h,int pts):name(n),health(h),points(pts){}

    string  getname()
        {return name;}
    int gethealth()
        {return health;}
    int getpoints()
        {return points;}
    Fighter fight(Fighter,Fighter);
};
Fighter Fighter::fight(Fighter f1,Fighter f2)
{
    srand(time(0));
    Fighter f;
    bool turn=false,visit[SIZE];
    string w,enter_word;
    int indx,tries=0,total_tries=0,max_tries=10;

    for(int j=0;j<SIZE;j++) visit[j]=0;

    gotoxy(2,2);
    Above_horizental();
    gotoxy(2,3);
    Line();
    gotoxy(77,3);
    Line();
    gotoxy(2,4);
    below_Horizental();
    gotoxy(35,3);
    cout<<"FIGHT FOR WORD"<<endl<<endl;

start:
    while(f1.health!=0 || f2.health!=0)
    {
        system("pause");
        system("cls");

        gotoxy(2,2);
        Above_horizental();
        gotoxy(2,3);
        Line();
        gotoxy(77,3);
        Line();
        gotoxy(2,4);
        below_Horizental();
        gotoxy(35,3);
        cout<<"FIGHT FOR WORD"<<endl<<endl;

    	if(turn==false)
		{
			cout<<"          "<<f1.name<<"'s Turn\n";
        random1:
    		indx=rand()%SIZE;
        	w=w_words[indx];

        	if(visit[indx]==0)
        	{
        		visit[indx]=1;
                cout<<"Word is: >>>>"<<w<<"<<<<"<<endl;
                cout<<"Correct this word\n";
                cin>>enter_word;
                cout<<endl;

                if(enter_word==c_words[indx])
                {
                    cout<<"Correct\n";
                    f1.points+=2;
                    cout<<f1.name<<"'s Points = "<<f1.points<<endl<<endl;
                    turn=true;
                }
                else
                {
                    cout<<"<==Wrong==>\n";
                    f1.health--;
                    cout<<f1.name<<"'s Health is : "<<f1.health<<endl;
                    cout<<"Correct Word is: "<<c_words[indx]<<endl<<endl;
                    turn=true;
                }
            }
            else goto random1;
        }
		else
		{
			cout<<"          "<<f2.name<<"'s Turn\n";
        random2:
			indx=rand()%SIZE;
        	w=w_words[indx];
        	if(visit[indx]==0)
        	{
                visit[indx]=1;
                cout<<"Word is: <<<<"<<w<<">>>>\n";
                cout<<"Correct this word\n";
                cin>>enter_word;

                if(enter_word==c_words[indx])
                {
                    cout<<"Correct\n";
                    f2.points+=2;
                    if(f1.health==0) f2.health=0;
                    cout<<f2.name<<"'s Points = "<<f2.points<<endl<<endl;
                    turn=false;

                    if(f1.health==0 && f2.points>f1.points) return f2;
                    else if(f2.health==0 && f1.points>f2.points) return f1;
                }
                else
                {
                    cout<<"<==Wrong==>\n";
                    f2.health--;
                    if(f2.health==0) f1.health=0;
                    cout<<f2.name<<"'s Health : "<<f2.health<<endl;
                    cout<<"Correct Word is: "<<c_words[indx]<<endl<<endl;
                    turn=false;

                    if(f1.health==0 && f2.points>f1.points) return f2;
                    else if(f2.health==0 && f1.points>f2.points) return f1;
                }
            }
		else goto random2;
    	}
    	tries++;
    	total_tries++;

    	cout<<f1.name<<"'s    Points = "<<f1.points<<"     Health = "<<f1.health<<endl;
    	cout<<f2.name<<"'s    Points  = "<<f2.points<<"    Health = "<<f2.health<<endl<<endl;
    	cout<<"TOTAL TRIES = "<<tries<<endl;

        if(total_tries!=SIZE)
        {
            if(f1.health==f2.health && f1.points==f2.points && tries==max_tries) tries=0;
            else if(tries==max_tries && f1.points>f2.points) return f1;
            else if(tries==max_tries && f1.points<f2.points) return f2;
        }
        else
        {
            if(f1.health==f2.health && f1.points==f2.points && total_tries==SIZE) return f;
            else if(f1.points>f2.points && total_tries==SIZE) return f1;
            else if(f1.points<f2.points && total_tries==SIZE) return f2;
        }
	}
        if(f1.points==f2.points && tries<SIZE && f1.health==0 && f2.health==0)
        {
            cout<<"Scores are equal both players got an extra health use wisely.\n";
            f1.health=f2.health=1;
            tries=0;
            goto start;
        }
        else if(f1.points>f2.points) return f1;
        else if(f1.points>f2.points) return f2;
}
int main()
{
    string a,b;
    int h=3,pts=0;
    char play;

    gotoxy(2,2);
	Above_horizental();
	gotoxy(2,3);
	Line();
	gotoxy(77,3);
	Line();
	gotoxy(2,4);
	below_Horizental();
	gotoxy(35,3);
	cout<<"FIGHT FOR WORD"<<endl;

    start:
    cout<<"\nEnter Y to Play, N to quit:";
    cin>>play;
    if(play=='N' || play=='n') goto over;
    else if(play=='Y' || play=='y')
	{
	    system("cls");

        gotoxy(2,2);
		Above_horizental();
		gotoxy(2,3);
		Line();
		gotoxy(77,3);
		Line();
		gotoxy(2,4);
		Line();
		gotoxy(77,4);
		Line();
		gotoxy(2,5);
		Line();
		gotoxy(77,5);
		Line();
		gotoxy(2,6);
		Line();
		gotoxy(77,6);
		Line();
		gotoxy(2,7);
		Line();
		gotoxy(77,7);
		Line();
		gotoxy(2,8);
		Line();
		gotoxy(77,8);
		Line();
		gotoxy(2,9);
		Line();
		gotoxy(77,9);
		Line();
		gotoxy(2,10);
		Line();
		gotoxy(77,10);
		Line();
		gotoxy(2,11);
		Line();
		gotoxy(77,11);
		Line();
		gotoxy(2,12);
		Line();
		gotoxy(77,12);
		Line();
		gotoxy(2,13);
		Line();
		gotoxy(77,13);
		Line();
		gotoxy(2,14);
		Line();
		gotoxy(77,14);
		Line();
		gotoxy(2,15);
		Line();
		gotoxy(77,15);
		Line();
		gotoxy(2,16);
		Line();
		gotoxy(77,16);
		Line();
		gotoxy(2,17);
		Line();
		gotoxy(77,17);
		Line();
		gotoxy(2,18);
		Line();
		gotoxy(77,18);
		Line();
		gotoxy(2,19);
		Line();
		gotoxy(77,19);
		Line();
		gotoxy(2,20);
		Line();
		gotoxy(77,20);
		Line();
		gotoxy(2,21);
		below_Horizental();

		gotoxy(35,3);
        cout<<"==GAME RULES==\n";
		gotoxy(28,1);
        cout<<"       FIGHT FOR WORD"<<endl;
		gotoxy(4,6);
        cout<<"1) Total words are :"<<SIZE<<endl;
		gotoxy(4,8);
        cout<<"2) Each player will get randomly generated words\n";
		gotoxy(4,10);
        cout<<"3) Once a word is displayed will not be displayed again\n";
		gotoxy(4,12);
        cout<<"4) Each player will get <3> health\n";
		gotoxy(4,14);
        cout<<"5) Scoring highest points and keeping is live safe will b the WINNNER\n";
		gotoxy(4,16);
        cout<<"6) If points are equal and all healths are lost both will get an \n      EXTRA Health\n\n\n\n\n";

        system("pause");
        system("cls");

    	cout<<"Enter 1st Fighter name:\n            ";
    	cin>>a;
    	cout<<"Enter 2st Fighter name:\n            ";
    	cin>>b;
    	cout<<endl;
        again:

        Fighter f1(a,h,pts),f2(b,h,pts),f3;

    	cout<<f1.getname()<<" HEALTH "<<f1.gethealth()<<endl<<f2.getname()<<" HEALTH "<<f2.gethealth()<<endl<<endl;

		system("pause");
		system("cls");

    	f3=f3.fight(f1,f2);

    	system("pause");
		system("cls");

        gotoxy(2,2);
        Above_horizental();
        gotoxy(2,3);
        Line();
        gotoxy(77,3);
        Line();
        gotoxy(2,4);
        below_Horizental();
        gotoxy(35,3);
        cout<<"FIGHT FOR WORD"<<endl<<endl;

        if(f3.getpoints()==0)
        {
            char ag;
            cout<<"All words used press Y to start again :";
            cin>>ag;
            cout<<endl;
            if(ag=='y' || ag=='Y') goto again;
            else goto endd;
        }
        else
        {
            cout<<"\nWINNER IS: ====="<<f3.getname()<<"====="<<endl<<"Points = "<<f3.getpoints()<<endl<<endl;
            cout<<"\t    \\O/   \n";
            cout<<"\t     | \n";
            cout<<"\t    /\\  HURRY \n";
            cout<<"\tTTTTTTTTTTTTTTTT\n";
        }
    }
    else
    {
		system("cls");

        gotoxy(2,2);
		Above_horizental();
		gotoxy(2,3);
		Line();
		gotoxy(77,3);
		Line();
		gotoxy(2,4);
		below_Horizental();
		gotoxy(35,3);
		cout<<"FIGHT FOR WORD"<<endl<<endl;

        cout<<"Enter Again\n";
        goto start;
    }
    over:
    endd:
	cout<<"<<<<<<<<<<<<<<<<<GAME OVER>>>>>>>>>>>>>>>>>>>>>>>";
return 0;
}
void gotoxy(int x, int y)
	{
		COORD coord;
		coord.X = x;
		coord.Y = y;
		SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
	}
	void Above_horizental()
	{
		char prev = ' ';
		prev = cout.fill((char)205);
		cout << (char)201 << setw(74) << "" << (char)187 << endl;
		cout.fill(prev);
	}
	void Line()
	{
	cout << (char)186<< endl;
	}
	void below_Horizental()
	{
		char prev = ' ';
		prev = cout.fill((char)205);
		cout << (char)200 << setw(74) << "" << (char)188 << endl;
		cout.fill(prev);
	}
