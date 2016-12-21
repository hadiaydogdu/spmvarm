#include "method.h"
#include "profiler.h"
#include "csrByNZAnalyzer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#if defined(__linux__) && defined(__x86_64__)
#include "lib/Target/X86/MCTargetDesc/X86BaseInfo.h"
#else
#include "lib/Target/ARM/MCTargetDesc/ARMBaseInfo.h"
#endif

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"

using namespace llvm;
using namespace spMVgen;
using namespace std;

extern unsigned int NUM_OF_THREADS;

class CSRbyNZCodeEmitter : public SpMVCodeEmitter {
private:
  NZtoRowMap *rowByNZs;
  unsigned long baseValsIndex, baseRowsIndex;
  
  void dumpSingleLoop(unsigned long numRows, unsigned long rowLength);
  
protected:
  virtual void dumpPushPopHeader();
  virtual void dumpPushPopFooter();
  
public:
  CSRbyNZCodeEmitter(NZtoRowMap *rowByNZs, unsigned long baseValsIndex,
                     unsigned long baseRowsIndex, llvm::MCStreamer *Str, unsigned int partitionIndex);
  
  void emit();
};


CSRbyNZ::CSRbyNZ(Matrix *csrMatrix):
  SpMVMethod(csrMatrix), analyzer(csrMatrix) {
  // do nothing
}

// Return a matrix to be used by CSRbyNZ
// rows: row indices, sorted by row lengths
// cols: indices of elements,
//       sorted according to the order used in rows array
// vals: values as usual,
//       sorted according to the order used in rows array
Matrix* CSRbyNZ::getMatrixForGeneration() {
  START_OPTIONAL_TIME_PROFILE(getCSRbyNZInfo);
  vector<NZtoRowMap> *rowByNZLists = analyzer.getRowByNZLists();
  END_OPTIONAL_TIME_PROFILE(getCSRbyNZInfo);
  
  START_OPTIONAL_TIME_PROFILE(matrixConversion);
  int *rows = new int[csrMatrix->n];
  int *cols = new int[csrMatrix->nz];
  double *vals = new double[csrMatrix->nz];

  vector<MatrixStripeInfo> &stripeInfos = csrMatrix->getStripeInfos();

#pragma omp parallel for
  for (int t = 0; t < NUM_OF_THREADS; ++t) {
    auto &rowByNZList = rowByNZLists->at(t);
    int *rowsPtr = rows + stripeInfos[t].rowIndexBegin;
    int *colsPtr = cols + stripeInfos[t].valIndexBegin;
    double *valsPtr = vals + stripeInfos[t].valIndexBegin;
    
    for (auto &rowByNZ : rowByNZList) {
      unsigned long rowLength = rowByNZ.first;
      for (int rowIndex : *(rowByNZ.second.getRowIndices())) {
        *rowsPtr++ = rowIndex;
        int k = csrMatrix->rows[rowIndex];
        for (int i = 0; i < rowLength; i++, k++) {
          *colsPtr++ = csrMatrix->cols[k];
          *valsPtr++ = csrMatrix->vals[k];
        }
      }
    }
  }
  END_OPTIONAL_TIME_PROFILE(matrixConversion);

  return new Matrix(rows, cols, vals, csrMatrix->n, csrMatrix->nz);
}

void CSRbyNZ::dumpAssemblyText() {
  START_OPTIONAL_TIME_PROFILE(getMatrix);
  this->getMatrix(); // this is done to measure the cost of matrix prep.
  vector<NZtoRowMap> *rowByNZLists = analyzer.getRowByNZLists(); // this has zero cost, because already calculated by getMatrix()
  vector<MatrixStripeInfo> &stripeInfos = csrMatrix->getStripeInfos();
  END_OPTIONAL_TIME_PROFILE(getMatrix);

  START_OPTIONAL_TIME_PROFILE(emitCode);
  // Set up code emitters
  vector<CSRbyNZCodeEmitter> codeEmitters;
  for (unsigned i = 0; i < rowByNZLists->size(); i++) {
    NZtoRowMap &rowByNZs = rowByNZLists->at(i);
    codeEmitters.push_back(CSRbyNZCodeEmitter(&rowByNZs, stripeInfos[i].valIndexBegin, stripeInfos[i].rowIndexBegin, Str, i));
  }
  
#pragma omp parallel for
  for (int threadIndex = 0; threadIndex < NUM_OF_THREADS; ++threadIndex) {
    codeEmitters[threadIndex].emit();
  }
  END_OPTIONAL_TIME_PROFILE(emitCode);
}


CSRbyNZCodeEmitter::CSRbyNZCodeEmitter(NZtoRowMap *rowByNZs, unsigned long baseValsIndex,
                                       unsigned long baseRowsIndex, llvm::MCStreamer *Str, unsigned int partitionIndex):
rowByNZs(rowByNZs), baseValsIndex(baseValsIndex), baseRowsIndex(baseRowsIndex) {
  this->DFOS = createNewDFOS(Str, partitionIndex);
}

void CSRbyNZCodeEmitter::emit() {
  dumpPushPopHeader();


//  for (auto &rowByNZ : *rowByNZs) {
//    unsigned long rowLength = rowByNZ.first;
//    dumpSingleLoop(rowByNZ.second.getRowIndices()->size(), rowLength);
//  }  
    dumpSingleLoop(100, 4);
 
  dumpPushPopFooter();
  emitRETInst();
}
  
void CSRbyNZCodeEmitter::dumpPushPopHeader() {
#if defined(__linux__) && defined(__x86_64__)
  // rows is in %rdx, cols is in %rcx, vals is in %r8
  emitPushPopInst(X86::PUSH64r,X86::R8);
  emitPushPopInst(X86::PUSH64r,X86::R9);
  emitPushPopInst(X86::PUSH64r,X86::R10);
  emitPushPopInst(X86::PUSH64r,X86::R11);
  emitPushPopInst(X86::PUSH64r,X86::RAX);
  emitPushPopInst(X86::PUSH64r,X86::RBX);
  emitPushPopInst(X86::PUSH64r,X86::RCX);
  emitPushPopInst(X86::PUSH64r,X86::RDX);
  
  emitLEAQInst(X86::RDX, X86::RDX, (int)(sizeof(int) * baseRowsIndex));
  emitLEAQInst(X86::RCX, X86::RCX, (int)(sizeof(int) * baseValsIndex));
  emitLEAQInst(X86::R8, X86::R8, (int)(sizeof(double) * baseValsIndex));
#else
   emitPushArmInst(ARM::R4);
   emitPushArmInst(ARM::R5);
   emitPushArmInst(ARM::R6);
#endif

}

void CSRbyNZCodeEmitter::dumpPushPopFooter() {
#if defined(__linux__) && defined(__x86_64__)
  emitPushPopInst(X86::POP64r, X86::RDX);
  emitPushPopInst(X86::POP64r, X86::RCX);
  emitPushPopInst(X86::POP64r, X86::RBX);
  emitPushPopInst(X86::POP64r, X86::RAX);
  emitPushPopInst(X86::POP64r, X86::R11);
  emitPushPopInst(X86::POP64r, X86::R10);
  emitPushPopInst(X86::POP64r, X86::R9);
  emitPushPopInst(X86::POP64r, X86::R8);
#else
emitPopArmInst(ARM::R4);
emitPopArmInst(ARM::R5);
emitPopArmInst(ARM::R6);
#endif
}

void CSRbyNZCodeEmitter::dumpSingleLoop(unsigned long numRows, unsigned long rowLength) {


//printf("arm:r15 :%d \n",ARM::R15); 
  unsigned long labeledBlockBeginningOffset = 0;
  
  //mov     r3, #0
  // 0:       00 30 a0 e3          mov     r3, #0
  emitMOVArmInst(ARM::R3, 0x0);
  emitMOVWArmInst(ARM::R9, numRows*4);
   

 // printf("mov     r3, #0 \n");

  //vmov.i32        d16, #0x0
  //10 00 c0 f2
  emitVMOVI32ArmInst(ARM::D16, 0x0);
 // printf("vmov.i32        d16, #0x0 \n");

  //.align 16, 0x90
//  emitCodeAlignment(16);
  //.LBB0_1:
   labeledBlockBeginningOffset = DFOS->size();
//printf("labeled blog %d\n",labeledBlockBeginningOffset);
  
  // done for a single row
  for(int i = 0 ; i < rowLength ; i++){
//1c:
   //	    vldr    d17, [r2, i*8]
   //edd21b00  //0
   //edd21b02  //8
   //edd21b04  //16
		  emitVLDRArmInst(ARM::D17, ARM::R2, i);
//		  printf("vldr    d17, [r2, #%d]\n",i*8);

   //		ldr     r5, [r1, i*4]
   //e5915000  // 0
   //e5915004
   //e5915008
		  emitLDROffsetArmInst(ARM::R5, ARM::R1, i); // i*4 bir registera atilip ldr register kullanilabilir. bu sekilde en son cols +=400 buraya eklenebilir. 
//		  printf("ldr     r5, [r1, #%d]\n",i*4);

   //     add     r5, lr, r5, lsl #3
   //e08e5185
		  emitADDRegisterArmInst(ARM::R5, ARM::R6, ARM::R5, 3);
//		  printf("add     r5, lr, r5, lsl #3 \n");

   //		vldr    d20, [r5]
   //edd54b00
		  emitVLDRArmInst(ARM::D20, ARM::R5, 0x0);
//		  printf("vldr    d20, [r5]\n");

   //     vmul.f64        d17, d17, d20
   //ee611ba4
		  emitVMULArmInst(ARM::D17, ARM::D17, ARM::D20);
//		  printf("vmul.f64        d17, d17, d20 \n");

   //	    vadd.f64        d16, d16, d17
   //ee700ba1

		  emitVADDArmInst(ARM::D16, ARM::D16, ARM::D17);
//		  printf("vadd.f64        d16, d16, d17 \n");

  }
  
  //ldr     r5, [r0, r3]
  //e7905003
  emitLDRRegisterArmInst(ARM::R5, ARM::R0, ARM::R3);
 // printf("ldr     r5, [r0, r3] \n");

  //add     r5, r4, r5, lsl #3
  //e0845185
  emitADDRegisterArmInst(ARM::R5, ARM::R4, ARM::R5, 3);
 // printf("add     r5, r4, r5, lsl #3 \n");

  //vldr    d18, [r5]
  //edd52b00
  emitVLDRArmInst(ARM::D18, ARM::R5, 0x0);
 // printf("vldr d18, [r5] \n");

  //vadd.f64        d16, d18, d16
  //ee720ba0
  emitVADDArmInst(ARM::D16, ARM::D18, ARM::D16);
 // printf("vadd.f64        d16, d18, d16 \n");

  //vstr    d16, [r5]
  //edc50b00
  emitVSTRArmInst(ARM::D16, ARM::R5);
  //printf("vstr    d16, [r5] \n");

  //add     r2, r2, #24
  //e2822018
  emitADDOffsetArmInst(ARM::R2, ARM::R2, 8*rowLength);
 
// printf ("add     r2, r2, #24 \n");

  //add     r1, r1, #12
  //e281100c
  emitADDOffsetArmInst(ARM::R1, ARM::R1, 4*rowLength);
 // printf("add     r1, r1, #12 \n");

  //add     r3, r3, #4
  //e2833004
  emitADDOffsetArmInst(ARM::R3, ARM::R3, 4);//add yerine ldr offseti kullansak?
 // printf("add     r3, r3, #4 \n");

  //cmp     r3, #400
  //e3530e19
  emitCMPRegisterArmInst(ARM::R3, ARM::R9);
 // printf("cmp     r3, #%d \n",numRows);

  //bne     .LBB0_1
  //  88:   1affffe3        bne     1c <_Z4spmvPiS_Pd+0x1c>
  emitBNEArmInst(labeledBlockBeginningOffset);
//  printf("bne     .LBB0_1\n");
}
