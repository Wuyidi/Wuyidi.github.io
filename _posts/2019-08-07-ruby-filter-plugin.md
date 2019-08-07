---
layout:     post
title:      "Ruby Filter Plugin"
subtitle:   "随心所欲处理日志文件"
date:       2019-08-07
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - logstash
---

> Using [logstash-filter-ruby](https://github.com/logstash-plugins/logstash-filter-ruby), you can use all the power of Ruby string manipulation to parse an extoic regular expression, an incomplete date format, write to a file, or even make a web service call.



### Logstash Installation

If you haven’t installed Logstash already, refer to the [official instructions here](https://www.elastic.co/guide/en/logstash/current/installing-logstash.html).

Assuming you have installed Logstash at "/home/logstash", create "/home/logstash/config/ruby-logstash.conf"

```ruby
input {
  file {
    path => "/path/to/log/file"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}
filter {
  ruby {
  	init => ""
    code => ""
  }
  
}
output {
	stdout {
  	codec => rebuydebug 
  }  
}
```



### Event Object

Referenced by [Event API](https://www.elastic.co/guide/en/logstash/current/event-api.html)

#### Get API

The getter is a read-only access of field-based data in an Event.

**Syntax:** `event.get(field)`

**Returns:** Value for this field or nil if the field does not exist. Returned values could be a string, numeric or timestamp scalar value.

#### Set API

This API can be used to mutate data in an Event.

**Syntax:** `event.set(field, value)`

**Returns:** The current Event after the mutation, which can be used for chainable calls.

### Sample Event

```
[10:48:15.303479] [dbg] ucid(11908) read msg (type=900006,seqno=10017,fid=1,cid=N_9FCF63B6-7D39-5504-FA78-D45034BA925B,datasize=244,dest=amigw3rd)
[10:48:15.303591] [dbg] ipc(7992) read msg(type=900007,seqno=10017,fid=2,cid=N_9FCF63B6-7D39-5504-FA78-D45034BA925B,datasize=197,dest=)
```

### Logstash Config

#### init 

> Any code to execute at logstash startup-time

```ruby
class CacheManager
  @@cache = Hash.new
  def addMsg(uid, msg)
  	@@cache[uid] = msg
  end
  def getMsg(uid)
    if @@cache.include?(uid)
    	return @@cache[uid]
    else
      return nil
    end
  end
  def removeMsg(uid)
    @@cache.delete(uid)
  end
end

class EventManager
  def initialize(cacheMgr, logdate)
    @cacheMgr = cacheMgr
    @logdate = logdate
  end
	def getyymmdd()
    t = @logdate[0..9].split(/\-/)
    return t.to_a.map(&:to_i)
  end
  def gethhmmss(str)
		return str.split(/\:|\./).map(&:to_i)
  end
  def coverttime(value)
    t = Time.mktime(*value)
  end
  def splitMsg(msg)
    @valid = 0
    str1 = msg.split
    str3 = msg.slice(/msg.*,dest/)
    if str3 != nil
      str2 = str3.split(/\,|\=|\(|\)/)
      if str2.length == 12
        @datetime = converttime(getyymmdd() + gethhmmss(str1[0]))
        @seqno = str2[4]
        @cid = str2[8]
        @msgType = str2[2]
        @fid = str2[6].to_i
      else
        @valid = 1
      end
    else
      @valid = 1
    end
  end
  def handlerequest()
    uid = @cid + @seqno
    @cacheMgr.addMsg(uid, @value)
  end
  def handleresponse()
    uid = @cid + @seqno
    tmp = @cacheMgr.getMsg(uid)
    @cacheMgr.remove(uid)
    if tmp == nil
      return false
    else
      @sTime = tmp['datetime']
      @eTime = @datetime
      @msgType = tmp['msgType']
      @cid = tmp['cid']
      @seqno = tmp['seqno']
      return true
    end
  end
  def handleEvent(event)
  	splitMsg(event)
    @value = Hash.new
    @value['datetime'] = @datetime
    @value['seqno'] = @seqno
    @value['cid'] = @cid
    @value['msgType'] = @msgType
    if @valid = 0
      if @fid == 1
        handlerequest()
      else
        handleresponse()
      end
    end
  end
  end
  def dateTime
    @datetime
  end
  def seqno
    @seqno
  end
  def cid()
    @cid
  end
  def msgType()
    @msgType
  end
  def fid()
    @fid
  end
  def sTime()
    @sTime
  end
  def eTime()
    @eTime
  end
end
```



#### code

> The code to execute for every event. You will have an `event` variable available that is the event itself. See the [Event API](https://www.elastic.co/guide/en/logstash/current/event-api.html) for more information.

```ruby
cacheMgr = CacheManager.new
eventMgr = EventManager.new
msg = event.get('message')
timestamp = event.get('@timestamp').to_s
eventMgr.splitMsg(msg)
if eventMgr.handleEvent(msg)
  dimensions = Hash.new
  normalFields = Hash.new
  measures = Hash.new
  lantency = (eventMgr.eTime - eventMgr.sTime)*1000
  dimensions = {
  	'msgType' => eventMgr.msgType,
    'sTime' => eventMgr.sTime,
    'eTime' => eventMgr.eTime
 	}
  normalFields = {
    'fid' => eventMgr.fid,
    'seqno' => eventMgr.seqno,
    'cid' => eventMgr.cid,
    'logdate' => timestamp
  }
```

