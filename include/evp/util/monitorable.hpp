#pragma once
#ifndef EVP_UTIL_MONITORABLE_H
#define EVP_UTIL_MONITORABLE_H

#include <map>
#include <tr1/functional>

#include <clip.hpp>

namespace evp {
using namespace clip;

class Monitorable {
 public:
  typedef i32 ListenerID;

 private:
  ListenerID curID_;
  f32 progress_;
  std::map< ListenerID, std::tr1::function<void (f32)> > listeners_;

 protected:
  void setProgress(f32 progress) {
    progress_ = progress;
    
    CurrentQueue().finish();
    
    std::map< ListenerID, std::tr1::function<void (f32)> >::iterator it, end;
    for (it = listeners_.begin(), end = listeners_.end(); it != end; ++it)
      it->second(progress_);
  }

 public:
  Monitorable() : curID_(0), progress_(0) {}
 
  ListenerID addProgressListener(std::tr1::function<void (f32)> listener) {
    listeners_[++curID_] = listener;
    return curID_;
  }
  
  void removeProgressListener(ListenerID listenerID) {
    listeners_.erase(listenerID);
  }
  
  f32 progress() { return progress_; }
};

}

#endif