// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda-data.h"

corpus* read_data(char* data_filename)
{
    FILE *fileptr;
    int length, count, word, n, nd, nw;
    corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    c->docs = 0;
    c->num_terms = 0;
    c->num_docs = 0;
    fileptr = fopen(data_filename, "r");
    if(fileptr == NULL){
        printf("error in reading file");
        return;
    }
    nd = 0; nw = 0;
    char t_id[256]; 
    while ((fscanf(fileptr, "%s", t_id) != EOF))
//    while (!feof(fileptr))
    {
   //     c->docs[nd].id =  malloc(sizeof(char)*256);;
        //fscanf(fileptr, "%s", t_id);
        //printf(" %d, %s \n", nd,t_id);
       // fscanf(fileptr, "%10d", &length);
	c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
       // fscanf(fileptr, "%s", c->docs[nd].id);
        fscanf(fileptr, "%10d", &length);
	c->docs[nd].length = length;
	c->docs[nd].total = 0;
	c->docs[nd].words = malloc(sizeof(int)*length);
	c->docs[nd].counts = malloc(sizeof(int)*length);
        sprintf(c->docs[nd].id,"%s",t_id);        
   //     c->docs[nd].id = t_id;
      //  printf(" %d, %s \n", nd, c->docs[nd].id);
	for (n = 0; n < length; n++)
	{
	    fscanf(fileptr, "%10d:%10d", &word, &count);
  //          printf(" %d:%d ", word, count);
	    word = word - OFFSET;
	    c->docs[nd].words[n] = word;
	    c->docs[nd].counts[n] = count;
	    c->docs[nd].total += count;
	    if (word >= nw) { nw = word + 1; }
	}

  //      printf("%d,legth=%d file=%s\n",nd,length,data_filename);
	nd++;
    }
    fclose(fileptr);
    c->num_docs = nd;
  // for(n=0; n<nd;n++){
  //      printf(" %d, %s \n", n, c->docs[n].id);
  //  }
    c->num_terms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    return(c);
}

int max_corpus_length(corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->num_docs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}
