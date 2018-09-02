//tagWEATHERRADAR雷达信息的结构
typedef struct tagWEATHERRADAR
{
	char cFileType[16];						//3830数据标识(“CINRADC”)
											//973试验标识(“973”)
	char cCountry[30];						//国家名
	char cProvince[20];						//省名
	char cStation[40];						//站名	
	char cStationNumber[10];				//区站号
	char cRadarType[20];					//雷达型号
	char cLongitude[16];					//天线所在经度
	char cLatitude[16];						//天线所在纬度
	long lLongitudeValue;					//具体经度
	long lLatitudeValue;					//具体纬度
	long lHeight;							//天线海拔高度
	short sMaxAngle;						//地物阻挡最大仰角
	short sOptAngle;						//最佳观测仰角
	unsigned char ucSYear1;					//观测开始时间的年千百位(19-20)
	unsigned char ucSYear2;					//观测开始时间的年十个位(00-99)
	unsigned char ucSMonth;				//观测开始时间的月(1-12)
	unsigned char ucSDay;					//观测开始时间的日(1-31)
	unsigned char ucSHour;					//观测开始时间的时(0-23)
	unsigned char ucSMinute;				//观测开始时间的分(0-59)
	unsigned char ucSSecond;				//观测开始时间的秒(0-59)
	unsigned char ucTimeFrom;				//时间来源 0-计算机时钟(1天内未对时)
											//		   1-计算机时钟(1天内已对时)
											//		   2-GPS
											//		   3-其它
	unsigned char ucEYear1;					//观测结束时间的年千百位(19-20)
	unsigned char ucEYear2;					//观测结束时间的年十个位(00-99)
	unsigned char ucEMonth;				//观测结束时间的月(1-12)
	unsigned char ucEDay;					//观测结束时间的日(1-31)
	unsigned char ucEHour;					//观测结束时间的时(0-23)
	unsigned char ucEMinute;				//观测结束时间的分(0-59)
	unsigned char ucESecond;				//观测结束时间的秒(0-59)
	unsigned char ucScanMode;				//扫描方式  1-RHI  10-PPI和ZPPI  
											//		   1XX=VPPI(XX为扫描圈数)
	unsigned long ulSmilliSecond;			//以微秒为单位表示的秒的小数位
	unsigned short usRHIA;					//RHI所在的方位角(0.01度为单位) PPI和
											//VPPI时为FFFF
	short sRHIL;			//RHI所在的最低仰角(0.01度为单位) PPI和
							//VPPI时为FFFF
	short sRHIH;							//RHI所在的最高仰角(0.01度为单位) PPI和
											//VPPI时为FFFF
	unsigned short usEchoType;				//回波类型  0x405a-Z  0x406a-V 
											// 0x407a-W  0x408a-ZVW三要素
	unsigned short usProdCode;				//数据类型  0x8001-PPI数据 
											// 		   0x8002-RHI数据 
											// 		   0x8003-VPPI数据  
											//		   0x8004-单强度PPI数据  
											//		   0x8005-CAPPI数据
	unsigned char ucCalibration;				//标校状态  0-无  1-自动
												//  2-1星期内人工  3-1月内人工
	unsigned char remain1[3];				//保留字
	unsigned char remain2[660];				//保留字,放VPPISCANPARAMETER数据
											//对PPI和RHI文件:只有1层结构数据
											//对VPPI文件:有N层结构数据
	long lAntennaG;						//天线增益(0.001dB)
	long lPower;							//峰值功率(瓦)
	long lWavelength;						//波长(微米)
	unsigned short usBeamH;				//垂直波束宽度(秒)
	unsigned short usBeamL;				//水平波束宽度(秒)
	unsigned short usPolarization;				//极化状态 0-水平 1-垂直 2-双偏振
												// 3-圆偏振 4-其它
	unsigned short usLogA;					//对数动态范围(0.01dB)
	unsigned short usLineA;					//线性动态范围(0.01dB)
	unsigned short usAGCP;					//AGC延迟量(微秒)
	unsigned short usFreqMode;				//频率方式	1-单重复频率
											//  2-双重复频率3:2  3-双重复频率4:3
	unsigned short usFreqRepeat;				//重复频率
	unsigned short usPPPPulse;				//PPP脉冲数
	unsigned short usFFTPoint;				//FFT间隔点数
	unsigned short usProcessType;			//信号处理方式	1-PPP
											//	2-全程FFT	3-单程FFT
	unsigned char ucClutterT;				//杂波消除阀值(即STC)
	char cSidelobe;						//第一旁瓣(dB)
	unsigned char ucVelocityT;				//速度门限
	unsigned char ucFilderP;					//地物消除方式	0-无	
												//	1-IIR滤波器1	2-IIR滤波器2
												//	3-IIR滤波器3	4-IIR滤波器4
	unsigned char ucNoiseT;					//噪声消除阀值(即强度门限)
	unsigned char ucSQIT;					//SQI门限
	unsigned char ucIntensityC;				//DVIP强度值估算采用的通道
											// 1-对数通道 2-线性通道
	unsigned char ucIntensityR;				//强度值估算是否距离订正
											// 0-无(dB) 1-已订正(dBZ)
	unsigned char ucCalNoise;				//噪声系数标定值
	unsigned char ucCalPower;				//发射功率标定值
	unsigned char ucCalPulseWidth;			//脉冲宽度标定值
	unsigned char ucCalWorkFreq;			//工作频率标定值
	unsigned char ucCalLog;					//对数斜率标定值
	char remain3[92];						//保留字
	unsigned long int liDataOffset;			//数据偏移地址
}WEATHERRADAR;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//tagVPPISCANPARAMETER仰角层的结构
typedef struct tagVPPISCANPARAMETER
{
	unsigned short usMaxV;			//最大可测速度(厘米/秒)
	unsigned short usMaxL;			//最大可测距离(10米)
	unsigned short usBindWidth;		//库长(米)
	unsigned short usBinNumber;		//每径向库数
	unsigned short usRecordNumber;	//本圈径向数
	unsigned short usArotate;		//本圈转速(0.01度/秒)
	unsigned short usPrf1;			//本圈第一次重复频率(0.1Hz)
	unsigned short usPrf2;			//本圈第二次重复频率(0.1Hz)
	unsigned short usSpulseW;		//本圈脉宽(微秒)
	short		   usAngle;			//仰角(0.01度)
	unsigned char cSweepStatus;		//1=单要素	2=三要素(单重频)	3=三要素(双重频)
	unsigned char cAmbiguousp;		//0=无软件退模糊	1=软件退模糊
}VPPISCANPARAMETER;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
