---
layout:     post
title:      "如何实现filter功能"
subtitle:   ""
date:       2017-11-11
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - Android
---

## Android SerachView & Adapter

### Filter ArrayList

[Filtering ListView with custom (object) adapter](http://stackoverflow.com/questions/5780289/filtering-listview-with-custom-object-adapter)

### Adding Search Functionality to ListView

[Search Functionality Tutorial](http://www.androidhive.info/2012/09/android-adding-search-functionality-to-listview/)

###  Android ListView Custom Filter and Filterable interface

[Filterable Tutorial](http://www.survivingwithandroid.com/2012/10/android-listview-custom-filter-and.html)

为了实现List View 中 Search 功能 有下列几种模式:

* 通过加载menu中的SearchView, 绑定`setOnQueryTextLinstener`来实现

  ```java
  searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
              @Override
              public boolean onQueryTextSubmit(String query) {
                  return false;
              }

              @Override
              public boolean onQueryTextChange(String newText) {
                  adapter.getFilter().filter(newText.toString());
                  listView.setAdapter(adapter);
                  createAdapter(monsterList);
                  return true;
              }
          });
  ```

  ​

* 通过监听EditText, 实现search功能

  ```java
  editText.addTextChangedListener(new TextWatcher() {

  	@Override
  	public void onTextChanged(CharSequence s, int start, int before, int count) {
  		System.out.println("Text ["+s+"]");
  		aAdpt.getFilter().filter(s.toString());
  	}

  	@Override
  	public void beforeTextChanged(CharSequence s, int start, int count,int after) {

  	}

  	@Override
  	public void afterTextChanged(Editable s) {
  	}
  });
  ```

* 通过创建searchable.xml文件并且修改AndroidManifest.xml来使用系统提供的search功能

  AndroidMainfest.xml

  ```xml
  <activity android:name=".activity.SearchMonsterActivity">

              <intent-filter>
                  <category android:name="android.intent.category.LAUNCHER"/>
                  <action android:name="android.intent.action.SEARCH" />
              </intent-filter>
              <meta-data android:name="android.app.searchable"
                  android:resource="@xml/searchable">
              </meta-data>
  </activity>
  ```

  searchable.xml

  ```xml
  <searchable
      xmlns:android="http://schemas.android.com/apk/res/android"
      android:label="@string/app_name"
      android:hint="Search Monster">
  </searchable>
  ```

  ​

search 方法实现

```java
public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_activity_search, menu);
        setSearchView(menu);
        SearchManager searchManager = (SearchManager) 
        getSystemService(Context.SEARCH_SERVICE);
        SearchView searchView = (SearchView)
        menu.findItem(R.id.searchView).getActionView();
        SearchableInfo searchableInfo = 
          searchManager.getSearchableInfo(getComponentName());
        searchView.setSearchableInfo(searchableInfo);
        return super.onCreateOptionsMenu(menu);
    }

    private void setSearchView(Menu menu) {
        MenuItem item = menu.getItem(0);
        searchView = new SearchView(this);
        searchView.setIconifiedByDefault(false);
        searchView.setQueryHint("search monster");
        searchView.setSubmitButtonEnabled(true);
        item.setActionView(searchView);
    }
```



本文通过对`baseAdapter implements Filterable `来实现搜索栏filter的功能

```java
// Implement Filterable method
    @Override
    public Filter getFilter() {
       return new Filter() {
            @Override
            protected FilterResults performFiltering(CharSequence constraint) {
                FilterResults results = new FilterResults();
                // We implement here the filter logic
                if (constraint == null || constraint.length() == 0) {
                    // No filter implemented we return all the list
                    results.values = monsterList;
                    results.count = monsterList.size();
                } else {
                    // We perform filtering operation
                    ArrayList<Monster> searchList = new ArrayList<>();
                    for (int i = 0; i < monsterList.size(); i++) {
                        int index = monsterList.get(i).getName().toUpperCase().indexOf(constraint.toString().toUpperCase());

                        if (index != -1) {
                            searchList.add(monsterList.get(i));
                        }

                    }
                    results.values = searchList;
                    results.count = searchList.size();
                }
                return results;
            }

            @Override
            protected void publishResults(CharSequence constraint, FilterResults results) {
                // Now we have to inform the adapter about the new list filtered
                if (results.count == 0)
                    notifyDataSetInvalidated();
                else {
                    monsterList = (ArrayList<Monster>) results.values;
                    notifyDataSetChanged();
                }

            }
        };
    }
```




