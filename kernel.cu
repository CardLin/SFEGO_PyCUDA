
__global__ void GMEMD_gradient(float *data, float *diff, float *direct,
						int *list_x, int *list_y, float *list_deg,
						int list_len, int width, int height) {
	//kernel index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

	if(x<width && y<height){
		//init variable
		float pos_avg=0,neg_avg=0;
		int pos_count=0,neg_count=0,weight=0,start,end,mid,max_weight=0,max_token,dist_token;
		int tx,ty,otx,oty,stx,sty,etx,ety;
		float current,target,otarget,starget,etarget;
		
		//mid point
		current=data[y*width+x];

		//init direction (sliding windows, init weight)
		start=0;
		end=list_len/2+1;
		mid=end/2+1;
		weight=0;
		
		max_token=mid;
		max_weight=weight;	//weight can be negative, just searching for largest weight
		
		//calc direct
		for(int i=0;i<list_len;++i){
			stx=x+list_x[start];
			sty=y+list_y[start];
			etx=x+list_x[end];
			ety=y+list_y[end];
			
			//check start target of sliding window
			if( (stx>=0 && stx<width) && (sty>=0 && sty<height) ){
				starget=data[sty*width+stx];
			}
			else starget=-1;
			
			//check end target of sliding window (assume it is opposite of start target)
			if( (etx>=0 && etx<width) && (ety>=0 && ety<height) ){
				etarget=data[ety*width+etx];
			}
			else etarget=-1;
			
			//compare start and end to the middle point
			if(starget>current && etarget>current){
				//both is larger than middle point
				if(starget>etarget) --weight;
				else ++weight;
			}
			else{
				if(starget>current) --weight;
				if(etarget>current) ++weight;
			}
			
			//update max_weight
			if(weight>max_weight){
				max_token=mid;
				max_weight=weight;
			}
			else if(weight==max_weight){
				tx=x+list_x[max_token];
				ty=y+list_y[max_token];
				if( (tx>=0 && tx<width) && (ty>=0 && ty<height) ){
					starget=data[ty*width+tx];
				}
				else starget=0;
				
				tx=x+list_x[mid];
				ty=y+list_y[mid];
				if( (tx>=0 && tx<width) && (ty>=0 && ty<height) ){
					etarget=data[ty*width+tx];
				}
				else etarget=0;
				
				if(starget < etarget){
					max_token=mid;
				}
			}
			
			//move sliding window
			start=(start+1)%list_len;
			end=(end+1)%list_len;
			mid=(mid+1)%list_len;
		}
		
		//calculate diff (magnitude)
		for(int i=0;i<list_len;++i){
			tx=x+list_x[i];
			ty=y+list_y[i];
			if( (tx>=0 && tx<width) && (ty>=0 && ty<height) ){
				target=data[ty*width+tx];
				if(i>max_token) dist_token=i-max_token;
				else dist_token=max_token-i;
				if(dist_token>list_len/2) dist_token=list_len-dist_token;
				
				if(dist_token>list_len/4){
					pos_avg+=target;
					++pos_count;
				}
				else{
					neg_avg+=target;
					++neg_count;
				}
			}
		}
		
		//finish diff (magnitude)
		if(pos_count){ pos_avg/=(float)pos_count; }
		else{ pos_avg=current; }
		if(neg_count){ neg_avg/=(float)neg_count; }
		else{ neg_avg=current; }
		diff[y*width+x]=pos_avg-neg_avg;
		
		//direct finish
		direct[y*width+x]=list_deg[max_token];
	}
}


__global__ void GMEMD_integral(float *result, float *diff, float *direct,
                        int *list_x, int *list_y, float *list_deg,
						int list_len, int width, int height) {
	
	//kernel index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	if(x<width && y<height){
		int tx,ty;
		result[y*width+x]=0;
		for(int i=0;i<list_len;++i){
			tx=x+list_x[i];
			ty=y+list_y[i];
			if( (tx>=0 && tx<width) && (ty>=0 && ty<height) ){
				result[y*width+x]+=cos(direct[ty*width+tx]-list_deg[i])*diff[ty*width+tx];
			}
		}
	}
}

