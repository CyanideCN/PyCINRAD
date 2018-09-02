struct RADARSITE {
	char   country[30];			//国家名，文本格式输入
	char   province[20];			//省名，文本格式输入
	char   station[40];			//站名，文本格式输入
	char   stationnumber[10];	//区站号，文本格式输入
	char   radartype[20];		//雷达型号，文本格式输入
	char   longitude[16];		//天线所在经度，文本格式输入.书写格式例：E 115°32′12″
	char   latitude[16];			//天线所在纬度，文本格式输入.书写格式例：N 35°30′15″
	long int  longitudevalue;	    //天线所在经度的数值，以1 / 100度为计数单位（十进制）东经（E）为正，西经（W）为负
	long int  lantitudevalue;		//天线所在纬度的数值，以1 / 100度为计数单位（十进制）北纬（N）为正，南纬（S）为负
	long int  height;			//天线的海拔高度以毫米为计数单位
	short   Maxangle;		//测站四周地物阻挡的最大仰角（以1 / 100度为计数单位）
	short   Opangle;		//测站的最佳观测仰角（地物回波强度<10dBz，以1 / 100	度为计数单位）
	short   MangFreq;		//速调管频率（通过此频率可计算雷达波长）
};

struct RADARPERFORMANCEPARAM {
	long int	 AntennaG;	//天线增益，以0.001dB为计数单位
	unsigned short  BeamH;		//垂直波束宽度，以1 / 100度为计数单位
	unsigned short  BeamL;		//水平波束宽度，以1 / 100度为计数单位
	unsigned char  polarizations;	//极化状况
	char   sidelobe;	//第一旁瓣计数单位：dB（注意：输入负号）
	long int  Power;		//雷达脉冲峰值功率，以瓦为计数单位
	long int  wavelength;	//波长，以微米为计数单位
	unsigned short  logA;	//对数接收机动态范围, 以0.01dB为计数单位
	unsigned short  LineA;	//线性接收机动态范围, 以0.01为计数单位
	unsigned short  AGCP;	//AGC延迟量，以微秒为计数单位
	unsigned char 	 clutterT;		//杂波消除阀值，计数单位0.01dB
	unsigned char 	 VelocityP;	//速度处理方式
	unsigned char 	 filderP;		//地物消除方式
	unsigned char	 noiseT;	//噪声消除阀值	（0 - 255）
	unsigned char 	 SQIT;	//SQI阀值，以厘米 / 秒为计数单位
	unsigned char 	 intensityC;//rvp强度值估算采用的通道
	unsigned char	intensityR;//强度估算是否进行了距离订正
};

struct LAYERPARAM {
	unsigned char  ambiguousp; 	//本层退模糊状态
	unsigned short	 Arotate;		//本层天线转速, 计数单位:0.01度 / 秒
	unsigned short	 Prf1;	//本层的第一种脉冲重复频率, 计数单位 : 1 / 10 Hz
	unsigned short	 Prf2;	//本层的第二种脉冲重复频率, 计数单位 : 1 / 10 Hz
	unsigned short	 spulseW;		//本层的脉冲宽度, 计数单位:	微秒
	unsigned short	 MaxV;	//本层的最大可测速度, 计数单位 : 厘米 / 秒
	unsigned short	 MaxL;		//本层的最大可测距离，以10米为计数单位
	unsigned short	 binWidth;	//本层数据的库长，以分米为计数单位
	unsigned short	 binnumber;	//本层每个径向的库数
	unsigned short	 recordnumber;	//本层径向数(记录个数)
	short	 Swangles;			//本层的仰角，计数单位	：1 / 100度
};

struct RADAROBSERVATIONPARAM {
	unsigned char 	 stype;		//扫描方式
	unsigned short 	 syear;		//观测记录开始时间的年
	unsigned char 	 smonth;		//观测记录开始时间的月（1 - 12）
	unsigned char 	 sday;		//观测记录开始时间的日（1 - 31）
	unsigned char 	 shour;		//观测记录开始时间的时（00 - 23）
	unsigned char 	 sminute;		//观测记录开始时间的分（00 - 59）
	unsigned char 	 ssecond;		//观测记录开始时间的秒（00 - 59）
	unsigned char 	 Timep;	//时间来源
	unsigned long int smillisecond; 		//秒的小数位（计数单位微秒）
	unsigned char 	 calibration;	//标校状态
	unsigned char 	 intensityI;	//强度积分次数（32 - 128）
	unsigned char 	 VelocityP;	//速度处理样本数（31 - 255）(样本数 - 1）
	struct LAYERPARAM LayerParam[30]; //各圈扫描状态设置
	unsigned short	 RHIA;		//作RHI时的所在方位角，计数单位为1 / 100度
	short	 RHIL;				//作RHI时的最低仰角，计数单位为1 / 100度
	short	 RHIH;				//作RHI时的最高仰角，计数单位为1 / 100度
	unsigned short 	 Eyear;		//观测结束时间的年
	unsigned char 	 Emonth;		//观测结束时间的月（1 - 12）
	unsigned char 	 Eday;		//观测结束时间的日（1 - 31）
	unsigned char 	 Ehour;		//观测结束时间的时（00 - 23）
	unsigned char 	 Eminute;		//观测结束时间的分（00 - 59）
	unsigned char 	 Esecond;		//观测结束时间的秒（00 - 59）
	unsigned char 	 Etenth;		//观测结束时间的1 / 100秒（00 - 59）
};

struct DATA {
	unsigned char m_dbz;	//强度值，实际dBZ = (m_dbz - 64) / 2;
	char	  m_vel;		//速度值,实际Velocity = 最大可测速度 * (m_Vel - 128) / 128。
	unsigned char  m_undbz;  //无订正强度值，实际dBZ = (m_undbz - 64) / 2;
	unsigned char  m_sw;		//谱宽值，计数单位为最大可测速度的256分之一，无回波时为零。
};

struct DATARECORD {
	unsigned short	 startaz, startel, endaz, endel; //角度算法为： 实际角度 = 角度值 * 360.0 / pow(2, 16);
	struct	 DATA	RawData[1024];
};

struct RADARDATAFILEHEADER {
	struct RADARSITE  RadarSiteInfo;
	struct RADARPERFORMANCEPARAM  RadarPerformanceInfo;
	struct RADAROBSERVATIONPARAM  RadarObservationInfo;
	char Reserved[163];
};
